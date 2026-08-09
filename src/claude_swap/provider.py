"""Third-party provider accounts: the ``env`` block of ``settings.json``.

Claude Code reaches a third-party provider (Amazon Bedrock, Bedrock/Mantle,
Google Vertex, Microsoft Foundry) with no credential at all — the whole
configuration is environment variables, which ``/setup-bedrock`` persists into
the ``env`` object of ``<config-home>/settings.json``. So a provider account
here *is* an env dict: activation writes it, capture reads it back.

Verified against the claude 2.1.223 bundle:

- The active provider is the first flag in :data:`PROVIDER_ORDER` that is set
  (claude's ``Mn()``).
- The wizard's env writer (``tff``) sets every key it does not need to
  ``undefined``. :data:`MANAGED_ENV_KEYS` is that clear-set: activation removes
  all of them before writing the new account's, or a leftover
  ``AWS_BEARER_TOKEN_BEDROCK`` would outrank a freshly activated static-key
  account.

A leaf module — imports only ``paths``/``settings``/``claude_locks``/
``exceptions``, never the switcher. Nothing here touches the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from claude_swap.claude_locks import proper_lockfile
from claude_swap.exceptions import ConfigError, ValidationError
from claude_swap.paths import get_claude_config_home
from claude_swap.settings import atomic_write_json

PROVIDER_CONFIG_VERSION = 1


@dataclass(frozen=True)
class ProviderSpec:
    """A provider's flag env var and its display label (as claude spells it)."""

    flag: str
    label: str


#: In claude's own resolution order (``Mn()``), so a block with several flags
#: set is read the way claude would run it.
PROVIDERS: dict[str, ProviderSpec] = {
    "bedrock": ProviderSpec("CLAUDE_CODE_USE_BEDROCK", "Amazon Bedrock"),
    "foundry": ProviderSpec("CLAUDE_CODE_USE_FOUNDRY", "Microsoft Foundry"),
    "anthropicAws": ProviderSpec(
        "CLAUDE_CODE_USE_ANTHROPIC_AWS", "Claude Platform on AWS"
    ),
    "anthropicGoogleCloud": ProviderSpec(
        "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD", "Claude Platform on Google Cloud"
    ),
    "mantle": ProviderSpec("CLAUDE_CODE_USE_MANTLE", "Amazon Bedrock (Mantle)"),
    "vertex": ProviderSpec("CLAUDE_CODE_USE_VERTEX", "Google Vertex AI"),
}

PROVIDER_ORDER = tuple(PROVIDERS)
PROVIDER_FLAGS = frozenset(spec.flag for spec in PROVIDERS.values())

#: Accepted CLI spellings, folded to the internal id.
_PROVIDER_ALIASES = {
    "anthropic-aws": "anthropicAws",
    "anthropic_aws": "anthropicAws",
    "anthropic-google-cloud": "anthropicGoogleCloud",
    "anthropic_google_cloud": "anthropicGoogleCloud",
    "google-cloud": "anthropicGoogleCloud",
}

#: Model pins, keyed as ``--pin-<tier>`` names them.
MODEL_PIN_KEYS = {
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
}

#: Everything cswap owns inside ``settings.json``'s ``env``, and therefore
#: everything it clears when switching away. A key not listed here would be
#: written on activation and never removed, outliving its account — which is why
#: ``--set`` refuses one rather than passing it through.
MANAGED_ENV_KEYS: frozenset[str] = PROVIDER_FLAGS | frozenset({
    # region / location
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "CLOUD_ML_REGION",
    "ANTHROPIC_GOOGLE_CLOUD_LOCATION",
    # credentials
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    # model selection
    *MODEL_PIN_KEYS.values(),
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION",
    # endpoints / project scoping
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "ANTHROPIC_BEDROCK_SERVICE_TIER",
    "ANTHROPIC_AWS_BASE_URL",
    "ANTHROPIC_AWS_WORKSPACE_ID",
    "ANTHROPIC_VERTEX_PROJECT_ID",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_GOOGLE_CLOUD_PROJECT",
    "ANTHROPIC_GOOGLE_CLOUD_WORKSPACE_ID",
    "ANTHROPIC_GOOGLE_CLOUD_BASE_URL",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
    "CLAUDE_CODE_SKIP_MANTLE_AUTH",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
    "CLAUDE_CODE_SKIP_ANTHROPIC_AWS_AUTH",
    "CLAUDE_CODE_SKIP_ANTHROPIC_GOOGLE_CLOUD_AUTH",
})

#: Prompted for rather than taken from argv, and flagged when exporting.
SECRET_ENV_KEYS = frozenset({
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_AWS_API_KEY",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "ANTHROPIC_FOUNDRY_AUTH_TOKEN",
})


def normalize_provider(name: str) -> str:
    """Fold a provider name to its internal id, or raise listing the valid ones."""
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


@dataclass(frozen=True)
class ProviderConfig:
    """A provider id plus the env block that activates it.

    Stored through the ordinary per-account credential store, so it inherits its
    0600 files and macOS Keychain backend — for a bearer-token or access-key
    account the block *is* secret material.
    """

    provider: str
    env: Mapping[str, str]

    @property
    def label(self) -> str:
        return PROVIDERS[self.provider].label

    def summary(self) -> str:
        """The label plus the keys it sets — never their values."""
        keys = sorted(set(self.env) - PROVIDER_FLAGS)
        return f"{self.label} — {', '.join(keys)}" if keys else self.label

    def to_dict(self) -> dict:
        return {
            "schemaVersion": PROVIDER_CONFIG_VERSION,
            "provider": self.provider,
            "env": dict(self.env),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def looks_like_provider_config(credentials: str | None) -> bool:
    """Whether a stored credential blob is a provider config, not a login.

    How every path (export, import, capture, usage sentinel) tells the kinds
    apart from bytes alone: an OAuth blob has no ``provider``, a raw API key is
    not JSON.
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
    return isinstance(data, dict) and data.get("provider") in PROVIDERS


def parse_provider_config(
    data: object, *, label: str = "provider config"
) -> ProviderConfig:
    """Validate a provider config (dict or JSON text). The one chokepoint.

    Checks only what would break activation: a provider claude knows, and keys
    cswap can also clear. Whether the keys form a *working* configuration is the
    provider's call — guessing here would refuse valid setups (an instance role,
    a credential file, an ambient chain), and a bad region or revoked token
    surfaces on the first request either way.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValidationError(f"{label} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValidationError(f"{label} must be a JSON object")

    provider = normalize_provider(str(data.get("provider") or ""))

    raw_env = data.get("env") or {}
    if not isinstance(raw_env, dict):
        raise ValidationError(f"{label}: env must be a JSON object")
    env = {
        validated_env_key(key, label): _validated_env_value(key, value, label)
        for key, value in raw_env.items()
        # A stored flag is redundant, not wrong — dropping it keeps parse
        # idempotent so a config survives its own round trip.
        if str(key).strip() not in PROVIDER_FLAGS
    }
    return ProviderConfig(
        provider=provider, env={PROVIDERS[provider].flag: "1", **env}
    )


def validated_env_key(key: object, label: str) -> str:
    """Accept only a key cswap can also clear (see :data:`MANAGED_ENV_KEYS`)."""
    name = str(key).strip()
    if name not in MANAGED_ENV_KEYS:
        raise ValidationError(
            f"{label}: '{name}' is not a provider variable claude-swap "
            "manages, so it could not be removed when switching away. "
            "Set it in settings.json directly if you want it always on."
        )
    return name


def _validated_env_value(key: object, value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label}: value for '{key}' must be a non-empty string")
    return value.strip()


def parse_env_assignment(assignment: str) -> tuple[str, str | None]:
    """Parse one ``--set`` argument into ``(key, value | None)``.

    ``None`` means no inline value, which the CLI turns into a prompt, so a
    secret need never appear in argv.
    """
    key, sep, value = assignment.partition("=")
    name = str(key).strip()
    if name in PROVIDER_FLAGS:
        raise ValidationError(
            f"--set: '{name}' is set from the provider itself; pass "
            "--provider instead of setting the flag."
        )
    name = validated_env_key(name, "--set")
    if not sep:
        return name, None
    return name, _validated_env_value(name, value, "--set")


def config_from_block(block: Mapping[str, str]) -> ProviderConfig | None:
    """The provider config a live env block represents; ``None`` if inactive.

    Carries every key verbatim: which one claude authenticates with is claude's
    business, so a captured config keeps behaving exactly as it did.
    """
    provider = next(
        (n for n in PROVIDER_ORDER if _truthy(block.get(PROVIDERS[n].flag))),
        None,
    )
    if provider is None:
        return None
    env = {PROVIDERS[provider].flag: "1"}
    env.update({
        key: value.strip()
        for key, value in block.items()
        if key not in PROVIDER_FLAGS and isinstance(value, str) and value.strip()
    })
    return ProviderConfig(provider=provider, env=env)


def _truthy(value: object) -> bool:
    """Claude's ``tr()``: set unless empty/0/false."""
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "0", "false")


# -- the settings.json splice ----------------------------------------------


def settings_path() -> Path:
    """Claude Code's user-scope ``settings.json``.

    Follows ``CLAUDE_CONFIG_DIR`` like the active credential store: both
    describe "the login this environment runs as".
    """
    return get_claude_config_home() / "settings.json"


def _load_settings_for_write() -> dict:
    """Read settings.json for a read-modify-write; a corrupt file raises.

    Degrading to ``{}`` would replace the user's whole configuration with a
    provider block.
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


def _managed_only(env: Mapping[str, object]) -> dict[str, str]:
    """The managed, string-valued subset of an ``env`` mapping."""
    return {
        key: value
        for key, value in env.items()
        if key in MANAGED_ENV_KEYS and isinstance(value, str) and value
    }


def read_live_block() -> dict[str, str]:
    """The managed env keys currently set in ``settings.json``, or ``{}``.

    Never raises: this feeds status and list renders.
    """
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    env = data.get("env")
    return _managed_only(env) if isinstance(env, dict) else {}


def live_provider_config() -> ProviderConfig | None:
    """The provider config the live ``settings.json`` activates, if any."""
    return config_from_block(read_live_block())


def apply_block(block: Mapping[str, str]) -> dict[str, str]:
    """Make ``block`` the managed portion of ``settings.json``'s ``env``.

    Replaces the managed keys wholesale — any not in ``block`` are deleted —
    leaving other settings and unmanaged env keys untouched. Returns the managed
    block that was live before, so a caller can restore it on rollback.

    Held under Claude Code's own ``settings.json.lock`` protocol so an
    in-session ``/config`` write cannot interleave, and written with
    ``atomic_write_json``, which writes *through* a symlink rather than
    detaching it (#192/#193) — a dotfiles-managed settings.json is common.
    """
    path = settings_path()
    with proper_lockfile(path.parent / (path.name + ".lock")):
        data = _load_settings_for_write()
        env = data.get("env")
        if env is not None and not isinstance(env, dict):
            raise ConfigError(
                f"{path}: 'env' is not a JSON object; fix it before "
                "switching to a third-party provider account"
            )
        env = dict(env or {})
        previous = _managed_only(env)
        for key in MANAGED_ENV_KEYS:
            env.pop(key, None)
        env.update(block)
        if env:
            data["env"] = env
        else:
            data.pop("env", None)  # claude strips default-valued keys; match it
        atomic_write_json(path, data)
        return previous


def clear_block() -> dict[str, str]:
    """Remove every managed env key. Returns what was live before."""
    return apply_block({})


def resolve_active_slot(
    live_block: Mapping[str, str],
    slot_blocks: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Which managed slot the live block belongs to, or ``None`` if unmanaged.

    Two slots holding an identical block are indistinguishable; the lowest wins
    so the answer is at least stable.
    """
    if not live_block:
        return None
    live = dict(live_block)
    return next(
        (
            slot
            for slot in sorted(slot_blocks, key=lambda s: (len(s), s))
            if dict(slot_blocks[slot]) == live
        ),
        None,
    )
