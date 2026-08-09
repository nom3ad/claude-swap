"""Third-party provider accounts: the ``env`` block of ``settings.json``.

Claude Code reaches Anthropic's first-party API with a *credential* (an OAuth
blob or a managed ``sk-ant-api…`` key), which is what ``credentials.py`` owns.
It reaches a third-party provider — Amazon Bedrock, Bedrock/Mantle, Google
Vertex, Microsoft Foundry — with no credential at all: the whole configuration
is **environment variables**, and Claude Code's own ``/setup-bedrock`` wizard
persists them into the ``env`` object of ``<config-home>/settings.json``.

Verified against the claude 2.1.223 bundle:

- The provider is chosen by a boolean flag env var, resolved in a fixed order
  (``Mn()``): ``CLAUDE_CODE_USE_BEDROCK`` → bedrock, ``…_FOUNDRY`` → foundry,
  ``…_ANTHROPIC_AWS``, ``…_ANTHROPIC_GOOGLE_CLOUD``, ``…_USE_MANTLE`` → mantle,
  ``…_USE_VERTEX`` → vertex. First flag set wins; see ``PROVIDER_ORDER``.
- AWS auth resolves in a fixed precedence: ``AWS_BEARER_TOKEN_BEDROCK`` (sent as
  ``Authorization: Bearer …``), else the static-key trio, else the ambient AWS
  provider chain (which honors ``AWS_PROFILE``, SSO, IMDS).
- The wizard's env writer (``tff``) sets every key it does *not* need to
  ``undefined``, so changing provider or auth method removes the previous one's
  keys. :data:`MANAGED_ENV_KEYS` is that same clear-set: activation deletes all
  of them before writing the new account's, or a leftover
  ``AWS_BEARER_TOKEN_BEDROCK`` would silently outrank a freshly activated
  static-key account.

This module is the single owner of the config ↔ env-block translation and of
the ``settings.json`` splice. It is deliberately a **leaf**: it imports only
``paths``/``fsutil``/``settings``/``claude_locks``/``exceptions`` and never the
switcher, mirroring how ``credentials.py`` stays a leaf collaborator.

Third-party accounts have no server identity, no refresh token, and no
subscription-quota endpoint, so nothing here talks to the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from claude_swap.claude_locks import proper_lockfile
from claude_swap.exceptions import ConfigError, ValidationError
from claude_swap.paths import get_claude_config_home
from claude_swap.settings import atomic_write_json

_logger = logging.getLogger("claude-swap")

#: Bumped only on a breaking change to the stored provider-config shape.
PROVIDER_CONFIG_VERSION = 1


@dataclass(frozen=True)
class ProviderSpec:
    """One third-party provider: its flag, label, and region variable.

    ``label`` is copied verbatim from the bundle's provider label map so cswap
    names a provider the way Claude Code's own UI does. ``region_key`` is the
    env var that provider takes its region/location from, or ``""`` when it has
    none (Foundry is addressed by resource, not region). ``aws`` marks the
    providers that authenticate through the AWS credential chain, which are the
    only ones with auth methods beyond ``environment``.
    """

    flag: str
    label: str
    region_key: str
    aws: bool


#: Providers in Claude Code's own resolution order (``Mn()``). A block with more
#: than one flag set resolves to the first entry here, exactly as claude does.
PROVIDERS: dict[str, ProviderSpec] = {
    "bedrock": ProviderSpec(
        "CLAUDE_CODE_USE_BEDROCK", "Amazon Bedrock", "AWS_REGION", True
    ),
    "foundry": ProviderSpec(
        "CLAUDE_CODE_USE_FOUNDRY", "Microsoft Foundry", "", False
    ),
    "anthropicAws": ProviderSpec(
        "CLAUDE_CODE_USE_ANTHROPIC_AWS", "Claude Platform on AWS", "AWS_REGION", True
    ),
    "anthropicGoogleCloud": ProviderSpec(
        "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
        "Claude Platform on Google Cloud",
        "ANTHROPIC_GOOGLE_CLOUD_LOCATION",
        False,
    ),
    "mantle": ProviderSpec(
        "CLAUDE_CODE_USE_MANTLE", "Amazon Bedrock (Mantle)", "AWS_REGION", True
    ),
    "vertex": ProviderSpec(
        "CLAUDE_CODE_USE_VERTEX", "Google Vertex AI", "CLOUD_ML_REGION", False
    ),
}

PROVIDER_ORDER = tuple(PROVIDERS)

#: CLI-friendly spellings accepted for a provider name, folded to the internal
#: id. Users type ``--provider anthropic-aws``; the stored config keeps claude's
#: own camelCase id so a config and a bundle string are directly comparable.
_PROVIDER_ALIASES = {
    "anthropic-aws": "anthropicAws",
    "anthropic_aws": "anthropicAws",
    "anthropic-google-cloud": "anthropicGoogleCloud",
    "anthropic_google_cloud": "anthropicGoogleCloud",
    "google-cloud": "anthropicGoogleCloud",
}

#: AWS auth methods, named as ``/setup-bedrock`` names them (``tff``'s switch).
#: ``environment`` stores nothing and lets the ambient AWS provider chain
#: resolve — the right choice for SSO, IMDS, or a container role.
AWS_AUTH_METHODS = ("bearer", "profile", "accessKey", "environment")

#: Non-AWS providers authenticate through their own SDK's ambient credentials
#: (ADC for Google, an API key for Foundry). cswap does not model those as
#: structured auth methods — they travel in the ``env`` passthrough — so those
#: providers accept only ``environment``.
_ENVIRONMENT_ONLY = ("environment",)

# -- the managed env keys ---------------------------------------------------
#
# Everything cswap owns inside settings.json's ``env``. Activation clears all
# of these before writing an account's own keys (see the module docstring on
# why a partial write is unsafe), so a key that belongs to a provider config
# MUST be listed here or switching away from that account would leave it live.

_PROVIDER_FLAGS = frozenset(spec.flag for spec in PROVIDERS.values())

_AUTH_KEYS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
})

_REGION_KEYS = frozenset({"AWS_REGION", "AWS_DEFAULT_REGION"}) | frozenset(
    spec.region_key for spec in PROVIDERS.values() if spec.region_key
)

#: Per-tier model pins, keyed the way ``--pin-<tier>`` and the wizard name them.
MODEL_PIN_KEYS = {
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
}

#: Provider-scoped keys cswap carries verbatim rather than modelling. Every one
#: is a key claude reads only under a third-party provider, so clearing them
#: when no provider is active cannot disturb a first-party login.
PASSTHROUGH_ENV_KEYS = frozenset({
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "ANTHROPIC_BEDROCK_SERVICE_TIER",
    "ANTHROPIC_AWS_BASE_URL",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_AWS_WORKSPACE_ID",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_GOOGLE_CLOUD_PROJECT",
    "ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID",
    "ANTHROPIC_GOOGLE_CLOUD_BASE_URL",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
    "CLAUDE_CODE_SKIP_MANTLE_AUTH",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
    "CLAUDE_CODE_SKIP_ANTHROPIC_AWS_AUTH",
    "CLAUDE_CODE_SKIP_ANTHROPIC_GOOGLE_CLOUD_AUTH",
})

MANAGED_ENV_KEYS: frozenset[str] = (
    _PROVIDER_FLAGS
    | _AUTH_KEYS
    | _REGION_KEYS
    | frozenset(MODEL_PIN_KEYS.values())
    | PASSTHROUGH_ENV_KEYS
)

#: Managed keys whose value is a secret: masked in every display surface, and
#: called out when exporting. Mirrors the bundle's own hidden-value set.
SECRET_ENV_KEYS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
})

#: Filename of the marker recording which slot the live block came from.
ACTIVE_MARKER_FILENAME = "active-provider.json"


def normalize_provider(name: str) -> str:
    """Fold a user-supplied provider name to its internal id.

    Accepts the internal ids, their kebab/snake spellings, and any casing.
    Raises :class:`ValidationError` listing the valid names — the CLI, capture,
    and import all resolve names through here so they reject identically.
    """
    raw = (name or "").strip()
    if not raw:
        raise ValidationError(
            f"provider is required; one of: {', '.join(PROVIDER_ORDER)}"
        )
    if raw in PROVIDERS:
        return raw
    folded = raw.lower()
    if folded in _PROVIDER_ALIASES:
        return _PROVIDER_ALIASES[folded]
    for known in PROVIDERS:
        if known.lower() == folded:
            return known
    raise ValidationError(
        f"unknown provider '{name}'; expected one of: {', '.join(PROVIDER_ORDER)}"
    )


def auth_methods_for(provider: str) -> tuple[str, ...]:
    """Auth methods a provider accepts (AWS providers get all four)."""
    return AWS_AUTH_METHODS if PROVIDERS[provider].aws else _ENVIRONMENT_ONLY


@dataclass(frozen=True)
class ProviderConfig:
    """A third-party provider configuration — one managed slot's material.

    Stored through the ordinary per-account credential store (so it inherits the
    0600 ``.enc`` files, the macOS Keychain backend, and ``.prev`` retention),
    because for ``bearer``/``accessKey`` accounts it *is* secret material.
    """

    provider: str
    auth_method: str
    region: str = ""
    bearer_token: str = ""
    aws_profile: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: str = ""
    #: Extra managed keys carried verbatim (see PASSTHROUGH_ENV_KEYS).
    env: Mapping[str, str] = field(default_factory=dict)
    #: tier -> model id (see MODEL_PIN_KEYS).
    model_pins: Mapping[str, str] = field(default_factory=dict)

    @property
    def spec(self) -> ProviderSpec:
        return PROVIDERS[self.provider]

    @property
    def label(self) -> str:
        """Human provider name, as Claude Code's own UI spells it."""
        return self.spec.label

    def describe_auth(self) -> str:
        """One-line description of how this account authenticates."""
        if self.auth_method == "bearer":
            return "bearer token"
        if self.auth_method == "profile":
            return f"AWS profile {self.aws_profile}"
        if self.auth_method == "accessKey":
            return f"access key {self.access_key_id}"
        return "ambient credentials"

    def to_dict(self) -> dict:
        """JSON form, omitting every field this auth method does not use."""
        out: dict = {
            "schemaVersion": PROVIDER_CONFIG_VERSION,
            "provider": self.provider,
            "authMethod": self.auth_method,
        }
        for key, value in (
            ("region", self.region),
            ("bearerToken", self.bearer_token),
            ("awsProfile", self.aws_profile),
            ("accessKeyId", self.access_key_id),
            ("secretAccessKey", self.secret_access_key),
            ("sessionToken", self.session_token),
        ):
            if value:
                out[key] = value
        if self.env:
            out["env"] = dict(self.env)
        if self.model_pins:
            out["modelPins"] = dict(self.model_pins)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def looks_like_provider_config(credentials: str | None) -> bool:
    """Whether a stored credential blob is a provider config rather than a login.

    Strict on purpose, and checked the same way everywhere a stored blob's kind
    has to be told from its bytes (export, import, capture): a provider config
    is a JSON object carrying a recognized ``provider`` and an ``authMethod``.
    An OAuth blob has neither, and a raw API key is not JSON at all.
    """
    if not credentials:
        return False
    text = credentials.strip()
    if not text.startswith("{"):
        return False
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("provider") in PROVIDERS
        and isinstance(data.get("authMethod"), str)
    )


def parse_provider_config(data: object, *, label: str = "provider config") -> ProviderConfig:
    """Validate a provider config (dict or JSON text) into a ProviderConfig.

    The single validation chokepoint: the CLI, live capture, and import all come
    through here, so a malformed config is rejected identically wherever it
    enters. Every rejection names the offending field — these are configs users
    type by hand.

    Raises:
        ValidationError: unknown provider, auth method not valid for it,
            an auth method missing its material, an unmanaged ``env`` key,
            an unknown model-pin tier, or a region on a provider without one.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValidationError(f"{label} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError(f"{label} must be a JSON object")

    provider = normalize_provider(str(data.get("provider") or ""))
    spec = PROVIDERS[provider]

    auth_method = str(data.get("authMethod") or "").strip()
    allowed = auth_methods_for(provider)
    if auth_method not in allowed:
        if spec.aws:
            raise ValidationError(
                f"{label}: unknown auth method '{auth_method}'; expected one "
                f"of: {', '.join(allowed)}"
            )
        raise ValidationError(
            f"{label}: {spec.label} authenticates through its own ambient "
            f"credentials, so authMethod must be 'environment' (got "
            f"'{auth_method}'). Pass provider-specific variables with --set."
        )

    def text(key: str) -> str:
        value = data.get(key, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValidationError(f"{label}: {key} must be a string")
        return value.strip()

    region = text("region")
    if region and not spec.region_key:
        raise ValidationError(
            f"{label}: {spec.label} has no region setting; address it with "
            f"--set ANTHROPIC_FOUNDRY_RESOURCE=… instead."
        )

    bearer_token = text("bearerToken")
    aws_profile = text("awsProfile")
    access_key_id = text("accessKeyId")
    secret_access_key = text("secretAccessKey")
    session_token = text("sessionToken")

    if auth_method == "bearer" and not bearer_token:
        raise ValidationError(f"{label}: bearer auth requires a bearerToken")
    if auth_method == "profile" and not aws_profile:
        raise ValidationError(f"{label}: profile auth requires an awsProfile")
    if auth_method == "accessKey" and not (access_key_id and secret_access_key):
        raise ValidationError(
            f"{label}: accessKey auth requires both accessKeyId and "
            "secretAccessKey"
        )
    if auth_method != "accessKey" and session_token:
        raise ValidationError(
            f"{label}: sessionToken applies to accessKey auth only"
        )

    # Drop material the chosen method doesn't use rather than storing a secret
    # that can never be activated (and would still be exported).
    if auth_method != "bearer":
        bearer_token = ""
    if auth_method != "profile":
        aws_profile = ""
    if auth_method != "accessKey":
        access_key_id = secret_access_key = session_token = ""

    raw_env = data.get("env") or {}
    if not isinstance(raw_env, dict):
        raise ValidationError(f"{label}: env must be a JSON object")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        env[_validated_env_key(key, label)] = _validated_env_value(key, value, label)

    raw_pins = data.get("modelPins") or {}
    if not isinstance(raw_pins, dict):
        raise ValidationError(f"{label}: modelPins must be a JSON object")
    pins: dict[str, str] = {}
    for tier, value in raw_pins.items():
        key = str(tier).strip().lower()
        if key not in MODEL_PIN_KEYS:
            raise ValidationError(
                f"{label}: unknown model tier '{tier}'; expected one of: "
                f"{', '.join(MODEL_PIN_KEYS)}"
            )
        pins[key] = _validated_env_value(tier, value, label)

    return ProviderConfig(
        provider=provider,
        auth_method=auth_method,
        region=region,
        bearer_token=bearer_token,
        aws_profile=aws_profile,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=session_token,
        env=env,
        model_pins=pins,
    )


def _validated_env_key(key: object, label: str) -> str:
    """Accept only a key cswap can also *clear* on the way out.

    An unmanaged key would be written on activation and then left behind
    forever, since :data:`MANAGED_ENV_KEYS` is what the clear-set iterates —
    it would outlive its account and leak into every other one.
    """
    name = str(key).strip()
    if name not in MANAGED_ENV_KEYS:
        raise ValidationError(
            f"{label}: '{name}' is not a provider variable claude-swap "
            "manages, so it could not be removed when switching away. "
            "Set it in settings.json directly if you want it always on."
        )
    if name in _PROVIDER_FLAGS:
        raise ValidationError(
            f"{label}: '{name}' is set from the provider itself; pass "
            "--provider instead of setting the flag."
        )
    return name


def _validated_env_value(key: object, value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label}: value for '{key}' must be a non-empty string")
    return value.strip()


def parse_env_assignment(assignment: str) -> tuple[str, str]:
    """Parse one ``KEY=VALUE`` from ``--set``. Raises on a bad key or shape."""
    if "=" not in assignment:
        raise ValidationError(
            f"--set expects KEY=VALUE, got '{assignment}'"
        )
    key, _, value = assignment.partition("=")
    name = _validated_env_key(key, "--set")
    return name, _validated_env_value(name, value, "--set")


def env_block_for(config: ProviderConfig) -> dict[str, str]:
    """The env block that activates ``config``.

    Mirrors the wizard's ``tff``: the provider flag, the region, exactly the
    keys the chosen auth method needs, then pins and passthrough. Only keys with
    a value appear — the *absence* of a managed key is how it gets cleared, and
    :func:`apply_block` deletes every managed key not present here.
    """
    spec = config.spec
    block: dict[str, str] = {spec.flag: "1"}
    if config.region and spec.region_key:
        block[spec.region_key] = config.region
    if config.auth_method == "bearer":
        block["AWS_BEARER_TOKEN_BEDROCK"] = config.bearer_token
    elif config.auth_method == "profile":
        block["AWS_PROFILE"] = config.aws_profile
    elif config.auth_method == "accessKey":
        block["AWS_ACCESS_KEY_ID"] = config.access_key_id
        block["AWS_SECRET_ACCESS_KEY"] = config.secret_access_key
        if config.session_token:
            block["AWS_SESSION_TOKEN"] = config.session_token
    for tier, model in config.model_pins.items():
        block[MODEL_PIN_KEYS[tier]] = model
    # Passthrough last so an explicit --set wins over a derived value (e.g. a
    # user pinning AWS_DEFAULT_REGION alongside --region).
    block.update(config.env)
    return block


def config_from_block(block: Mapping[str, str]) -> ProviderConfig | None:
    """Read a provider config out of a live env block; ``None`` when inactive.

    Resolution mirrors claude's ``Mn()``: the first flag in
    :data:`PROVIDER_ORDER` whose value is truthy wins, so a block with several
    flags is read exactly as claude would run it. Auth method is inferred with
    claude's own precedence — bearer, then static keys, then profile/ambient —
    so a captured account authenticates the way the live config already does.
    """
    provider = next(
        (
            name
            for name in PROVIDER_ORDER
            if _truthy(block.get(PROVIDERS[name].flag))
        ),
        None,
    )
    if provider is None:
        return None
    spec = PROVIDERS[provider]

    def value(key: str) -> str:
        raw = block.get(key)
        return raw.strip() if isinstance(raw, str) else ""

    bearer = value("AWS_BEARER_TOKEN_BEDROCK")
    access_key_id = value("AWS_ACCESS_KEY_ID")
    secret_access_key = value("AWS_SECRET_ACCESS_KEY")
    profile = value("AWS_PROFILE")

    if not spec.aws:
        auth_method = "environment"
    elif bearer:
        auth_method = "bearer"
    elif access_key_id and secret_access_key:
        auth_method = "accessKey"
    elif profile:
        auth_method = "profile"
    else:
        auth_method = "environment"

    region = value(spec.region_key) if spec.region_key else ""
    pins = {
        tier: value(key)
        for tier, key in MODEL_PIN_KEYS.items()
        if value(key)
    }
    # Whatever else is managed and set travels verbatim, minus the keys the
    # structured fields above already own for this provider/auth method.
    claimed = set(env_block_for(
        ProviderConfig(
            provider=provider,
            auth_method=auth_method,
            region=region,
            bearer_token=bearer,
            aws_profile=profile,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=value("AWS_SESSION_TOKEN"),
            model_pins=pins,
        )
    ))
    env = {
        key: value(key)
        for key in sorted(MANAGED_ENV_KEYS - claimed - _PROVIDER_FLAGS)
        if value(key)
    }
    return ProviderConfig(
        provider=provider,
        auth_method=auth_method,
        region=region,
        bearer_token=bearer,
        aws_profile=profile,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=value("AWS_SESSION_TOKEN") if auth_method == "accessKey" else "",
        env=env,
        model_pins=pins,
    )


def _truthy(value: object) -> bool:
    """Claude's ``tr()``: a var counts as set unless it is empty/0/false."""
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "0", "false")


def block_fingerprint(block: Mapping[str, str]) -> str:
    """Stable digest of an env block, for matching a live block to a slot.

    Order-independent and secret-free (a digest, never the values), so it can
    be logged and compared without exposing a bearer token.
    """
    canonical = json.dumps(
        {k: block[k] for k in sorted(block)}, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def redacted_block(block: Mapping[str, str]) -> dict[str, str]:
    """A display copy of a block with every secret value masked."""
    return {
        key: ("(hidden)" if key in SECRET_ENV_KEYS else value)
        for key, value in block.items()
    }


# -- the settings.json splice ----------------------------------------------


def settings_path() -> Path:
    """Claude Code's user-scope ``settings.json`` (``userSettings``).

    ``<config-home>/settings.json``, where config home follows
    ``CLAUDE_CONFIG_DIR`` — the same resolution
    :func:`paths.get_claude_config_home` implements and claude's own ``fct()``
    performs. Deliberately env-following, like the active credential store:
    both describe "the login this environment runs as".
    """
    return get_claude_config_home() / "settings.json"


def _settings_lock_dir() -> Path:
    return settings_path().parent / (settings_path().name + ".lock")


def _load_settings_for_write() -> dict:
    """Read settings.json for a read-modify-write; a corrupt file raises.

    Mirrors ``settings._read_raw_for_write``: degrading a malformed (and maybe
    hand-recoverable) settings file to ``{}`` would replace the user's whole
    configuration with a provider block.
    """
    path = settings_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"could not read {path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"{path} is not valid JSON ({e}); fix or delete it before "
            "switching to a third-party provider account"
        ) from e
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path} is not a JSON object; fix or delete it before "
            "switching to a third-party provider account"
        )
    return data


def read_live_block() -> dict[str, str]:
    """The managed env keys currently set in ``settings.json``.

    ``{}`` when the file is missing, unreadable, malformed, or simply has no
    managed keys — a read must never raise into a status/list render. Only
    string values are returned: claude coerces its env values from JSON, and a
    non-string there is a hand-edit cswap should report as absent rather than
    pretend to understand.
    """
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    env = data.get("env")
    if not isinstance(env, dict):
        return {}
    return {
        key: value
        for key, value in env.items()
        if key in MANAGED_ENV_KEYS and isinstance(value, str) and value
    }


def live_provider_config() -> ProviderConfig | None:
    """The provider config the live ``settings.json`` activates, if any."""
    return config_from_block(read_live_block())


def apply_block(block: Mapping[str, str]) -> dict[str, str]:
    """Make ``block`` the managed portion of ``settings.json``'s ``env``.

    Replaces the managed keys wholesale — every key in
    :data:`MANAGED_ENV_KEYS` not present in ``block`` is deleted — and leaves
    every other setting and every unmanaged env key untouched. Returns the
    managed block that was live *before* the write, so a caller can restore it
    on rollback (and so a no-op write can be detected).

    Two correctness properties, both borrowed rather than re-derived:

    - Held under a ``settings.json.lock`` taken with :func:`proper_lockfile`,
      the same directory-mutex protocol Claude Code uses, so an in-session
      ``/config`` write cannot interleave with the splice.
    - Written with :func:`settings.atomic_write_json`, which writes *through* a
      symlinked path instead of renaming over it (#192/#193). A dotfiles-managed
      ``~/.claude/settings.json`` is common, and detaching that link would
      silently strand every later change.
    """
    with proper_lockfile(_settings_lock_dir()):
        data = _load_settings_for_write()
        env = data.get("env")
        if env is not None and not isinstance(env, dict):
            raise ConfigError(
                f"{settings_path()}: 'env' is not a JSON object; fix it "
                "before switching to a third-party provider account"
            )
        env = dict(env or {})
        previous = {
            key: value
            for key, value in env.items()
            if key in MANAGED_ENV_KEYS and isinstance(value, str) and value
        }
        for key in MANAGED_ENV_KEYS:
            env.pop(key, None)
        env.update(block)
        if env:
            data["env"] = env
        else:
            # Claude Code strips default-valued keys; match it so clearing the
            # last provider leaves settings.json as it was beforehand.
            data.pop("env", None)
        atomic_write_json(settings_path(), data)
        return previous


def clear_block() -> dict[str, str]:
    """Remove every managed env key. Returns what was live before."""
    return apply_block({})


# -- the active-slot marker ------------------------------------------------


def marker_path(backup_root: Path) -> Path:
    return backup_root / ACTIVE_MARKER_FILENAME


def read_marker(backup_root: Path) -> str | None:
    """Slot recorded as the source of the live block, if any.

    A hint only — :func:`resolve_active_slot` verifies it by fingerprint and
    falls back to scanning, so a stale or hand-deleted marker degrades to a
    slower answer rather than a wrong one.
    """
    try:
        data = json.loads(marker_path(backup_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    slot = data.get("slot")
    return str(slot) if isinstance(slot, (str, int)) and str(slot) else None


def write_marker(backup_root: Path, slot: str, applied_at: str) -> None:
    """Record which slot the live block came from (best-effort).

    Never raises: the marker is a disambiguation hint, and losing it only costs
    a fingerprint scan. Failing an otherwise-complete switch over it would be
    the worse trade.
    """
    try:
        atomic_write_json(
            marker_path(backup_root),
            {"schemaVersion": 1, "slot": str(slot), "appliedAt": applied_at},
        )
    except Exception as e:  # pragma: no cover - best-effort
        _logger.warning("Could not record the active provider slot: %s", e)


def clear_marker(backup_root: Path) -> None:
    """Drop the marker (best-effort) — no provider account is active."""
    try:
        marker_path(backup_root).unlink(missing_ok=True)
    except OSError as e:  # pragma: no cover - best-effort
        _logger.warning("Could not clear the active provider marker: %s", e)


def resolve_active_slot(
    live_block: Mapping[str, str],
    slot_blocks: Mapping[str, Mapping[str, str]],
    marker_slot: str | None,
) -> str | None:
    """Which managed slot the live block is, or ``None`` when it is unmanaged.

    Fingerprint equality, not value comparison, so the match never handles a
    secret. The marker is consulted first and *verified*; a disagreement falls
    through to a scan, which makes an incorrect marker self-healing. Identical
    configs in two slots resolve to the marked one, then to the lowest slot
    number, so the answer is at least stable across calls.
    """
    if not live_block:
        return None
    live = block_fingerprint(live_block)
    if marker_slot is not None:
        candidate = slot_blocks.get(marker_slot)
        if candidate is not None and block_fingerprint(candidate) == live:
            return marker_slot
    for slot in sorted(slot_blocks, key=lambda s: (len(s), s)):
        if block_fingerprint(slot_blocks[slot]) == live:
            return slot
    return None


def with_region(config: ProviderConfig, region: str) -> ProviderConfig:
    """Copy of ``config`` with a different region (used by the CLI's overrides)."""
    return replace(config, region=region.strip())
