"""Tests for third-party provider accounts (Bedrock/Mantle/Vertex/Foundry).

These accounts are not credential-shaped: claude authenticates to a third-party
provider purely through the ``env`` block of ``settings.json`` (the same keys
``/setup-bedrock`` writes), with no OAuth blob, no ``oauthAccount``, no refresh
token, and no usage endpoint. Covers the config↔env translation and its
clear-set, validation, live capture, ``add-provider``, activation and rollback
in both switch directions, active-slot detection over a stale ``oauthAccount``,
the "no quota" usage display, the autoswitch opt-in, the ``cswap run`` guard,
and export/import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from claude_swap import provider
from claude_swap.exceptions import ConfigError, SessionError, ValidationError
from claude_swap.json_output import USAGE_PROVIDER, usage_fields
from claude_swap.models import Platform
from claude_swap.session import SessionManager
from claude_swap.switcher import SENTINEL_NOTES, ClaudeAccountSwitcher
from claude_swap.transfer import export_accounts, import_accounts

BEARER = "ABSKbedrock-access-token-example"
OTHER_BEARER = "ABSKanother-access-token-example"
OAUTH_JSON = json.dumps(
    {
        "claudeAiOauth": {
            "accessToken": "tok",
            "refreshToken": "rtok",
            "expiresAt": 99_999_999_999_999,
        }
    }
)

# The live Mantle block this feature was built against: a real /setup-bedrock
# style configuration, with a passthrough key the structured fields don't own.
MANTLE_BLOCK = {
    "CLAUDE_CODE_USE_MANTLE": "1",
    "AWS_REGION": "us-east-1",
    "AWS_BEARER_TOKEN_BEDROCK": BEARER,
    "ANTHROPIC_CUSTOM_HEADERS": "anthropic-workspace-id: proj_x",
}


def _switcher() -> ClaudeAccountSwitcher:
    s = ClaudeAccountSwitcher()
    s.platform = Platform.LINUX
    s._setup_directories()
    s._init_sequence_file()
    return s


def _config(name: str = "bedrock", **env) -> provider.ProviderConfig:
    """A provider config from plain env keys, via the real validator."""
    return provider.parse_provider_config({"provider": name, "env": env})


def _bearer_config(
    region: str = "us-east-1",
    *,
    name: str = "bedrock",
    bearer_token: str = BEARER,
) -> provider.ProviderConfig:
    """The common case: a provider addressed by region with a bearer token."""
    return _config(name, AWS_REGION=region, AWS_BEARER_TOKEN_BEDROCK=bearer_token)


def _write_settings(data: dict) -> Path:
    path = provider.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _read_settings() -> dict:
    return json.loads(provider.settings_path().read_text(encoding="utf-8"))


def _live_env() -> dict:
    return _read_settings().get("env", {})


def _seed_oauth_login(email: str = "me@example.com") -> None:
    """A credential-shaped login for the switch-direction tests."""
    home = Path.home()
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    (home / ".claude" / ".credentials.json").write_text(OAUTH_JSON, encoding="utf-8")
    (home / ".claude.json").write_text(
        json.dumps({
            "oauthAccount": {
                "emailAddress": email,
                "accountUuid": "uuid-1",
                "organizationUuid": "",
                "organizationName": "",
            },
            "hasCompletedOnboarding": True,
        }),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Provider registry and config ↔ env translation
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    def test_flags_match_claude_code(self):
        # Transcribed from the bundle's Mn()/label map; a rename here would
        # silently stop activating the provider.
        assert provider.PROVIDERS["bedrock"].flag == "CLAUDE_CODE_USE_BEDROCK"
        assert provider.PROVIDERS["mantle"].flag == "CLAUDE_CODE_USE_MANTLE"
        assert provider.PROVIDERS["vertex"].flag == "CLAUDE_CODE_USE_VERTEX"
        assert provider.PROVIDERS["foundry"].flag == "CLAUDE_CODE_USE_FOUNDRY"

    def test_resolution_order_matches_claude_code(self):
        # claude's Mn() checks bedrock, foundry, anthropicAws,
        # anthropicGoogleCloud, mantle, vertex — in that order.
        assert provider.PROVIDER_ORDER == (
            "bedrock", "foundry", "anthropicAws", "anthropicGoogleCloud",
            "mantle", "vertex",
        )

    @pytest.mark.parametrize(
        "spelling,expected",
        [
            ("bedrock", "bedrock"),
            ("BEDROCK", "bedrock"),
            ("anthropic-aws", "anthropicAws"),
            ("anthropic_google_cloud", "anthropicGoogleCloud"),
            ("anthropicAws", "anthropicAws"),
        ],
    )
    def test_normalize_provider(self, spelling, expected):
        assert provider.normalize_provider(spelling) == expected

    def test_unknown_provider_lists_the_valid_ones(self):
        with pytest.raises(ValidationError, match="unknown provider 'azure'"):
            provider.normalize_provider("azure")

    def test_every_provider_flag_is_managed(self):
        # A flag missing from the clear-set would survive a switch away.
        for spec in provider.PROVIDERS.values():
            assert spec.flag in provider.MANAGED_ENV_KEYS

    def test_every_secret_key_is_managed(self):
        assert provider.SECRET_ENV_KEYS <= provider.MANAGED_ENV_KEYS


class TestEnvBlock:
    def test_the_config_is_the_env_block(self):
        # Activation is a copy, not a translation: whatever keys were set are
        # exactly what lands in settings.json, plus the provider's own flag.
        config = _config(
            "bedrock", AWS_REGION="us-east-1", AWS_BEARER_TOKEN_BEDROCK=BEARER
        )
        assert config.env == {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
            "AWS_BEARER_TOKEN_BEDROCK": BEARER,
        }

    def test_the_flag_comes_from_the_provider_id(self):
        # So a config can never contradict itself about which provider it is.
        assert _config("mantle").env == {"CLAUDE_CODE_USE_MANTLE": "1"}
        assert _config("vertex").env == {"CLAUDE_CODE_USE_VERTEX": "1"}

    def test_a_stored_flag_is_dropped_not_rejected(self):
        # A round-tripped config carries its flag; re-deriving it keeps parse
        # idempotent instead of failing every config's own round trip.
        config = provider.parse_provider_config({
            "provider": "bedrock",
            "env": {"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"},
        })
        assert config.env == {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
        }

    @pytest.mark.parametrize(
        "name,env",
        [
            # Whatever a provider needs is just keys — no per-provider fields.
            ("bedrock", {"AWS_REGION": "eu-west-1", "AWS_PROFILE": "my-sso"}),
            ("bedrock", {
                "AWS_ACCESS_KEY_ID": "AKIA1",
                "AWS_SECRET_ACCESS_KEY": "s3cret",
                "AWS_SESSION_TOKEN": "tok",
            }),
            ("vertex", {
                "CLOUD_ML_REGION": "us-central1",
                "ANTHROPIC_VERTEX_PROJECT_ID": "my-proj",
            }),
            ("foundry", {"ANTHROPIC_FOUNDRY_RESOURCE": "my-resource"}),
            ("anthropicGoogleCloud", {"ANTHROPIC_GOOGLE_CLOUD_PROJECT": "p"}),
        ],
    )
    def test_any_provider_shape_is_carried_verbatim(self, name, env):
        config = _config(name, **env)
        assert config.env == {provider.PROVIDERS[name].flag: "1", **env}

    def test_model_pins_are_just_keys(self):
        config = _config(
            "bedrock",
            ANTHROPIC_DEFAULT_OPUS_MODEL="anthropic.claude-opus-4-8",
            ANTHROPIC_BEDROCK_SERVICE_TIER="priority",
        )
        assert config.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == (
            "anthropic.claude-opus-4-8"
        )
        assert config.env["ANTHROPIC_BEDROCK_SERVICE_TIER"] == "priority"

    def test_live_mantle_block_round_trips(self):
        # The real configuration this feature was built against.
        config = provider.config_from_block(MANTLE_BLOCK)
        assert config is not None
        assert config.provider == "mantle"
        assert config.env == MANTLE_BLOCK

    def test_round_trip_survives_json(self):
        config = provider.config_from_block(MANTLE_BLOCK)
        reparsed = provider.parse_provider_config(config.to_json())
        assert reparsed.env == MANTLE_BLOCK
        assert reparsed.provider == "mantle"

    def test_summary_names_keys_without_leaking_secrets(self):
        summary = provider.config_from_block(MANTLE_BLOCK).summary()
        assert "Amazon Bedrock (Mantle)" in summary
        assert "AWS_BEARER_TOKEN_BEDROCK" in summary  # the key
        assert BEARER not in summary  # never the value
        assert "CLAUDE_CODE_USE_MANTLE" not in summary  # implied by the label

    def test_no_provider_flag_is_not_a_provider(self):
        assert provider.config_from_block({"AWS_REGION": "us-east-1"}) is None
        assert provider.config_from_block({}) is None

    def test_falsy_flag_is_not_active(self):
        # claude's tr(): "0"/"false"/"" do not enable a provider.
        for value in ("0", "false", ""):
            assert provider.config_from_block(
                {"CLAUDE_CODE_USE_BEDROCK": value}
            ) is None

    def test_several_flags_resolve_like_claude(self):
        config = provider.config_from_block({
            "CLAUDE_CODE_USE_MANTLE": "1",
            "CLAUDE_CODE_USE_BEDROCK": "1",
        })
        assert config.provider == "bedrock"  # first in PROVIDER_ORDER

    def test_capture_does_not_interpret_the_keys(self):
        # Which key claude actually authenticates with is claude's business.
        # Capturing every key verbatim means a config with both a bearer token
        # and static keys keeps behaving exactly as it did before capture —
        # cswap has no auth-precedence model of its own to get wrong.
        block = {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_BEARER_TOKEN_BEDROCK": BEARER,
            "AWS_ACCESS_KEY_ID": "AKIA1",
            "AWS_SECRET_ACCESS_KEY": "s3cret",
        }
        assert provider.config_from_block(block).env == block

    def test_unmanaged_and_empty_keys_are_dropped_on_capture(self):
        config = provider.config_from_block({
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
            "AWS_PROFILE": "   ",
        })
        assert config.env == {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_REGION": "us-east-1",
        }


class TestValidation:
    """Only two rules, both because breaking them breaks activation itself.

    Whether the keys form a *working* configuration is deliberately not checked:
    the provider decides that, and a wrong guess here would refuse a valid
    setup. A missing region or revoked token surfaces on the first request,
    exactly as it would after ``/setup-bedrock``.
    """

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(ValidationError, match="unknown provider"):
            provider.parse_provider_config({"provider": "azure", "env": {}})

    def test_unmanaged_env_key_is_rejected(self):
        # The load-bearing one: an unmanaged key gets written on activation but
        # never cleared on the way out (MANAGED_ENV_KEYS is what the clear-set
        # iterates), so it would outlive its account and leak into every other.
        with pytest.raises(ValidationError, match="not a provider variable"):
            provider.parse_provider_config({
                "provider": "bedrock", "env": {"TOTALLY_UNRELATED": "x"},
            })

    def test_an_incomplete_config_is_accepted(self):
        # Static keys without their secret half is a real mistake — but it is
        # the provider's error to report, and refusing it here would also
        # refuse setups that work for reasons cswap cannot see (an instance
        # role, a credential file, an ambient chain).
        config = _config("bedrock", AWS_ACCESS_KEY_ID="AKIA1")
        assert config.env["AWS_ACCESS_KEY_ID"] == "AKIA1"

    def test_empty_values_are_rejected(self):
        with pytest.raises(ValidationError, match="must be a non-empty string"):
            _config("bedrock", AWS_REGION="")

    def test_non_object_env_is_rejected(self):
        with pytest.raises(ValidationError, match="env must be a JSON object"):
            provider.parse_provider_config({"provider": "bedrock", "env": "nope"})

    def test_malformed_json_is_rejected(self):
        with pytest.raises(ValidationError, match="not valid JSON"):
            provider.parse_provider_config("{nope")


class TestSetAssignment:
    def test_inline_value(self):
        assert provider.parse_env_assignment("AWS_PROFILE=dev") == (
            "AWS_PROFILE", "dev",
        )

    def test_bare_key_defers_to_a_prompt(self):
        # `--set KEY` with no value means "ask me", so a secret need never
        # appear in argv (and so in shell history).
        assert provider.parse_env_assignment("AWS_BEARER_TOKEN_BEDROCK") == (
            "AWS_BEARER_TOKEN_BEDROCK", None,
        )

    def test_stdin_sentinel_is_passed_through(self):
        assert provider.parse_env_assignment("AWS_SECRET_ACCESS_KEY=-") == (
            "AWS_SECRET_ACCESS_KEY", "-",
        )

    def test_unmanaged_key_is_rejected(self):
        with pytest.raises(ValidationError, match="not a provider variable"):
            provider.parse_env_assignment("TOTALLY_UNRELATED=x")

    def test_setting_a_provider_flag_by_hand_is_rejected(self):
        # Unlike a stored config (where the flag is redundant and dropped),
        # setting it by hand contradicts --provider, its only source.
        with pytest.raises(ValidationError, match="pass --provider"):
            provider.parse_env_assignment("CLAUDE_CODE_USE_VERTEX=1")

    def test_looks_like_provider_config(self):
        assert provider.looks_like_provider_config(_bearer_config().to_json())
        assert not provider.looks_like_provider_config(OAUTH_JSON)
        assert not provider.looks_like_provider_config("sk-ant-api03-xyz")
        assert not provider.looks_like_provider_config("")
        assert not provider.looks_like_provider_config(None)
        assert not provider.looks_like_provider_config('{"provider": "azure"}')


# ---------------------------------------------------------------------------
# The settings.json splice
# ---------------------------------------------------------------------------


class TestApplyBlock:
    def test_writes_the_block_and_preserves_other_settings(self, temp_home: Path):
        _write_settings({"effortLevel": "xhigh", "env": {"MY_OWN": "keep"}})
        previous = provider.apply_block(_bearer_config().env)

        assert previous == {}  # nothing managed was live before
        settings = _read_settings()
        assert settings["effortLevel"] == "xhigh"
        assert settings["env"]["MY_OWN"] == "keep"
        assert settings["env"]["CLAUDE_CODE_USE_BEDROCK"] == "1"

    def test_clears_sibling_keys_of_the_previous_auth_method(self, temp_home: Path):
        # The property /setup-bedrock's tff() gets by setting every unused key
        # to undefined: a leftover bearer token outranks static keys in claude's
        # own precedence, so a partial write would silently use the wrong one.
        provider.apply_block(_bearer_config().env)
        assert "AWS_BEARER_TOKEN_BEDROCK" in _live_env()

        provider.apply_block(_config(
            "bedrock",
            AWS_REGION="us-west-2",
            AWS_ACCESS_KEY_ID="AKIA1",
            AWS_SECRET_ACCESS_KEY="s3cret",
        ).env)
        env = _live_env()
        assert "AWS_BEARER_TOKEN_BEDROCK" not in env
        assert env["AWS_ACCESS_KEY_ID"] == "AKIA1"

    def test_switching_provider_clears_the_old_flag(self, temp_home: Path):
        provider.apply_block(_bearer_config(name="mantle").env)
        provider.apply_block(_bearer_config(name="bedrock").env)
        env = _live_env()
        assert "CLAUDE_CODE_USE_MANTLE" not in env
        assert env["CLAUDE_CODE_USE_BEDROCK"] == "1"

    def test_returns_the_previous_managed_block(self, temp_home: Path):
        _write_settings({"env": dict(MANTLE_BLOCK)})
        previous = provider.apply_block(_bearer_config().env)
        assert previous == MANTLE_BLOCK

    def test_clear_block_removes_only_managed_keys(self, temp_home: Path):
        _write_settings({
            "effortLevel": "xhigh",
            "env": {**MANTLE_BLOCK, "MY_OWN": "keep"},
        })
        provider.clear_block()
        settings = _read_settings()
        assert settings["env"] == {"MY_OWN": "keep"}
        assert settings["effortLevel"] == "xhigh"

    def test_clearing_the_last_key_drops_the_env_object(self, temp_home: Path):
        # claude strips default-valued keys; match it so a cleared provider
        # leaves settings.json as it was before.
        _write_settings({"effortLevel": "xhigh", "env": dict(MANTLE_BLOCK)})
        provider.clear_block()
        assert "env" not in _read_settings()

    def test_missing_settings_file_is_created(self, temp_home: Path):
        assert not provider.settings_path().exists()
        provider.apply_block(_bearer_config().env)
        assert _live_env()["CLAUDE_CODE_USE_BEDROCK"] == "1"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
    def test_writes_through_a_symlinked_settings_file(self, temp_home: Path):
        # A dotfiles-managed settings.json is common. Renaming over the link
        # would detach it and silently strand every later change (#192/#193);
        # atomic_write_json writes THROUGH it.
        real = temp_home / "dotfiles" / "settings.json"
        real.parent.mkdir(parents=True)
        real.write_text(json.dumps({"effortLevel": "xhigh"}), encoding="utf-8")
        link = provider.settings_path()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(real)

        provider.apply_block(_bearer_config().env)

        assert link.is_symlink(), "the symlink was replaced by a regular file"
        assert json.loads(real.read_text(encoding="utf-8"))["env"][
            "CLAUDE_CODE_USE_BEDROCK"
        ] == "1"

    def test_corrupt_settings_file_refuses_rather_than_clobbers(self, temp_home: Path):
        path = provider.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            provider.apply_block({"CLAUDE_CODE_USE_BEDROCK": "1"})
        assert path.read_text(encoding="utf-8") == "{not json"

    def test_non_object_env_refuses(self, temp_home: Path):
        _write_settings({"env": "nope"})
        with pytest.raises(ConfigError, match="'env' is not a JSON object"):
            provider.apply_block({"CLAUDE_CODE_USE_BEDROCK": "1"})

    def test_read_live_block_never_raises_on_bad_input(self, temp_home: Path):
        assert provider.read_live_block() == {}  # no file
        _write_settings({"env": "nope"})
        assert provider.read_live_block() == {}
        provider.settings_path().write_text("{oops", encoding="utf-8")
        assert provider.read_live_block() == {}

    def test_read_live_block_ignores_unmanaged_and_non_string_values(
        self, temp_home: Path
    ):
        _write_settings({"env": {"MY_OWN": "x", "AWS_REGION": 5, **MANTLE_BLOCK}})
        assert provider.read_live_block() == MANTLE_BLOCK | {"AWS_REGION": "us-east-1"}


class TestActiveSlotResolution:
    def test_matches_the_owning_slot(self):
        blocks = {"1": {"A": "1"}, "2": {"B": "2"}}
        assert provider.resolve_active_slot({"B": "2"}, blocks) == "2"

    def test_identical_configs_resolve_to_the_lowest_slot(self):
        assert provider.resolve_active_slot({"A": "1"}, {"2": {"A": "1"}, "1": {"A": "1"}}) == "1"

    def test_unmanaged_live_block_resolves_to_none(self):
        assert provider.resolve_active_slot({"A": "1"}, {"1": {"B": "2"}}) is None

    def test_empty_live_block_resolves_to_none(self):
        assert provider.resolve_active_slot({}, {"1": {}}) is None


# ---------------------------------------------------------------------------
# Registration: live capture and add-provider
# ---------------------------------------------------------------------------


class TestAddProviderAccount:
    def test_registers_a_slot_with_kind_provider(self, temp_home: Path, capsys):
        s = _switcher()
        slot = s.add_provider_account(_bearer_config())

        assert slot == "1"
        assert s.account_kind_for("1") == "provider"
        assert s._get_sequence_data()["accounts"]["1"]["email"] == (
            "bedrock-1@provider.local"
        )
        out = capsys.readouterr().out
        assert "Added" in out and "Amazon Bedrock" in out

    def test_stores_the_config_as_the_slot_credential(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        stored = json.loads(
            s._read_account_credentials("1", "bedrock-1@provider.local")
        )
        assert stored["provider"] == "bedrock"
        assert stored["env"]["AWS_BEARER_TOKEN_BEDROCK"] == BEARER

    def test_writes_a_synthesized_oauth_account_so_the_slot_is_switchable(
        self, temp_home: Path
    ):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        config = json.loads(s._read_account_config("1", "bedrock-1@provider.local"))
        assert config["oauthAccount"]["emailAddress"] == "bedrock-1@provider.local"
        assert s._account_is_switchable("1")

    def test_explicit_email_and_alias(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(
            _bearer_config(), email="work@example.com", alias="work"
        )
        record = s._get_sequence_data()["accounts"]["1"]
        assert record["email"] == "work@example.com"
        assert record["alias"] == "work"

    def test_invalid_config_is_rejected(self, temp_home: Path):
        s = _switcher()
        with pytest.raises(ValidationError, match="not a provider variable"):
            s.add_provider_account(
                {"provider": "bedrock", "env": {"TOTALLY_UNRELATED": "x"}}
            )
        assert s._get_sequence_data()["accounts"] == {}

    def test_refresh_in_place_updates_the_stored_config(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config(), email="p@example.com")
        s.add_provider_account(
            _bearer_config(bearer_token=OTHER_BEARER), email="p@example.com"
        )
        data = s._get_sequence_data()
        assert len(data["accounts"]) == 1
        stored = json.loads(s._read_account_credentials("1", "p@example.com"))
        assert stored["env"]["AWS_BEARER_TOKEN_BEDROCK"] == OTHER_BEARER

    def test_cross_kind_collision_with_an_oauth_slot(self, temp_home: Path):
        s = _switcher()
        s.add_account_from_token("sk-ant-oat01-abc", email="dup@example.com")
        with pytest.raises(ValidationError, match="already exists as an OAuth"):
            s.add_provider_account(_bearer_config(), email="dup@example.com")

    def test_oauth_token_rejected_when_email_is_a_provider(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config(), email="dup@example.com")
        with pytest.raises(
            ValidationError, match="already exists as an third-party provider"
        ):
            s.add_account_from_token("sk-ant-oat01-abc", email="dup@example.com")


class TestCaptureLiveProvider:
    def test_add_captures_the_live_block(self, temp_home: Path, capsys):
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()

        assert s.account_kind_for("1") == "provider"
        stored = s._provider_config_for("1", "mantle-1@provider.local")
        assert stored.env == MANTLE_BLOCK
        assert "Amazon Bedrock (Mantle)" in capsys.readouterr().out

    def test_capture_marks_the_slot_active(self, temp_home: Path):
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()
        assert s.active_provider_slot() == "1"
        assert s.current_account_number() == "1"
        assert s._get_sequence_data()["activeAccountNumber"] == 1

    def test_capture_wins_over_a_stale_oauth_account(self, temp_home: Path):
        # A Bedrock machine still has an oauthAccount from a previous login;
        # the live provider block is what claude actually authenticates with.
        _seed_oauth_login()
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()
        assert s.account_kind_for("1") == "provider"

    def test_second_add_registers_the_dormant_oauth_login(self, temp_home: Path):
        # The whole machine stays reachable: provider first (it is live), then
        # the login underneath it on the next run.
        _seed_oauth_login()
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()
        s.add_account()
        data = s._get_sequence_data()
        assert data["accounts"]["1"]["kind"] == "provider"
        assert data["accounts"]["2"]["email"] == "me@example.com"
        assert "kind" not in data["accounts"]["2"]

    def test_re_adding_an_identical_config_does_not_duplicate(self, temp_home: Path):
        # A managed provider is never re-captured (there is no token to
        # refresh), so the second add falls through to the OAuth login — and
        # with none present, says so rather than adding a duplicate slot.
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()
        with pytest.raises(ConfigError, match="already managed"):
            s.add_account()
        assert len(s._get_sequence_data()["accounts"]) == 1

    def test_an_explicit_slot_re_registers_the_live_config(self, temp_home: Path):
        # --slot is an explicit request to put it there, so it is honored even
        # though an identical config already occupies another slot.
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()
        s.add_account(slot=3, assume_yes=True)
        data = s._get_sequence_data()
        assert data["accounts"]["3"]["kind"] == "provider"

    def test_no_provider_and_no_login_still_reports_the_old_error(
        self, temp_home: Path
    ):
        s = _switcher()
        with pytest.raises(ConfigError, match="No active Claude account"):
            s.add_account()


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


class TestSwitchToProvider:
    def test_activating_writes_the_env_block(self, temp_home: Path):
        _seed_oauth_login()
        s = _switcher()
        s.add_account()  # slot 1: the OAuth login
        s.add_provider_account(_bearer_config())  # slot 2: the provider

        s.switch_to("2")

        assert _live_env()["CLAUDE_CODE_USE_BEDROCK"] == "1"
        assert _live_env()["AWS_BEARER_TOKEN_BEDROCK"] == BEARER
        assert s.active_provider_slot() == "2"
        assert s._get_sequence_data()["activeAccountNumber"] == 2

    def test_the_dormant_credential_is_left_in_place(self, temp_home: Path):
        # Deliberate: it is dormant while a provider flag is set, and keeping
        # it means switching back needs no re-login.
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")
        assert s._read_credentials() == OAUTH_JSON

    def test_the_outgoing_slot_is_backed_up(self, temp_home: Path):
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")
        assert s._read_account_credentials("1", "me@example.com") == OAUTH_JSON

    def test_active_slot_beats_a_stale_oauth_account(self, temp_home: Path):
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")

        # ~/.claude.json still names account 1 — the provider block wins.
        assert s._get_current_account() == ("me@example.com", "")
        assert s.current_account_number() == "2"
        info = {str(n): active for n, _, _, _, active, _, _ in s._build_accounts_info()}
        assert info == {"1": False, "2": True}

    def test_status_reports_the_provider_slot(self, temp_home: Path):
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")
        payload = s.status(json_output=True)
        assert payload["active"]["number"] == 2
        assert payload["active"]["usageStatus"] == "provider"

    def test_already_active_is_a_noop(self, temp_home: Path, capsys):
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()
        capsys.readouterr()

        result = s.switch_to("1", json_output=True)
        assert result["switched"] is False
        assert result["reason"] == "already-active"

    def test_force_rewrites_a_drifted_block(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        s.switch_to("1")
        # Someone hand-edited settings.json.
        provider.apply_block({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "xx"})

        s.switch_to("1", force=True)
        assert _live_env()["AWS_REGION"] == "us-east-1"

    def test_provider_to_provider_replaces_the_block(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config(name="mantle"))
        s.add_provider_account(
            _config("bedrock", AWS_REGION="eu-west-1", AWS_PROFILE="my-sso")
        )
        s.switch_to("1")
        s.switch_to("2")

        env = _live_env()
        assert "CLAUDE_CODE_USE_MANTLE" not in env
        assert "AWS_BEARER_TOKEN_BEDROCK" not in env
        assert env["AWS_PROFILE"] == "my-sso"
        assert s.active_provider_slot() == "2"

    def test_no_login_at_all_still_activates(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        s.switch_to("1")
        assert s.active_provider_slot() == "1"


class TestSwitchAwayFromProvider:
    def test_the_block_is_cleared(self, temp_home: Path):
        # It must go, or its provider flag keeps outranking the credential
        # being activated and the switch silently does not take effect.
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")
        assert _live_env()

        s.switch_to("1")
        assert provider.read_live_block() == {}
        assert s.active_provider_slot() is None
        assert s.current_account_number() == "1"

    def test_unmanaged_settings_survive(self, temp_home: Path):
        _seed_oauth_login()
        _write_settings({"effortLevel": "xhigh", "env": {"MY_OWN": "keep"}})
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")
        s.switch_to("1")

        settings = _read_settings()
        assert settings["effortLevel"] == "xhigh"
        assert settings["env"] == {"MY_OWN": "keep"}

    def test_round_trip_leaves_both_activatable(self, temp_home: Path):
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())

        s.switch_to("2")
        s.switch_to("1")
        s.switch_to("2")

        assert (
            s._provider_config_for("2", "bedrock-2@provider.local").env == _live_env()
        )
        assert s._read_account_credentials("1", "me@example.com") == OAUTH_JSON

    def test_an_unmanaged_block_is_also_cleared(self, temp_home: Path):
        # A block set up by /setup-bedrock (or left behind by removing the slot
        # that owned it) shadows the credential being activated just as much as
        # a managed one — without clearing it the switch would report success
        # while claude kept using the provider.
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        _write_settings({"env": dict(MANTLE_BLOCK)})
        assert s.active_provider_slot() is None  # nothing owns it

        s.switch_to("1")
        assert provider.read_live_block() == {}

    def test_an_unmanaged_block_defeats_the_already_active_noop(
        self, temp_home: Path
    ):
        # The slot IS the oauthAccount, so the no-op guard would short-circuit
        # and never reach the clear — leaving claude on the provider while cswap
        # reported the switch as done. (The self-switch still *reports*
        # already-active, since from == to; what matters is that it reconciled
        # rather than returning early.)
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.switch_to("1")
        _write_settings({"env": dict(MANTLE_BLOCK)})

        s.switch_to("1")
        assert provider.read_live_block() == {}

    def test_removing_the_live_provider_slot_clears_its_block(self, temp_home: Path):
        # Otherwise the configuration is stranded: still authenticating claude,
        # owned by no slot, and still holding a secret in plaintext after the
        # slot that stored it is gone.
        _write_settings({"effortLevel": "xhigh", "env": dict(MANTLE_BLOCK)})
        s = _switcher()
        s.add_account()

        s.remove_account("1", assume_yes=True)
        assert provider.read_live_block() == {}
        assert _read_settings()["effortLevel"] == "xhigh"

    def test_removing_an_inactive_provider_slot_leaves_the_block(
        self, temp_home: Path
    ):
        s = _switcher()
        s.add_provider_account(_bearer_config(name="mantle"))
        s.add_provider_account(_bearer_config(name="bedrock"))
        s.switch_to("1")
        live = provider.read_live_block()

        s.remove_account("2", assume_yes=True)
        assert provider.read_live_block() == live

    def test_a_failed_activation_restores_the_block(self, temp_home: Path, monkeypatch):
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())
        s.switch_to("2")
        live_before = provider.read_live_block()

        # Fail the credential write after the block has been cleared.
        def boom(_creds):
            raise RuntimeError("activation exploded")

        monkeypatch.setattr(s, "_write_credentials", boom)
        with pytest.raises(Exception):
            s.switch_to("1")

        assert provider.read_live_block() == live_before
        assert s.active_provider_slot() == "2"


class TestActiveSlotTracking:
    def test_follows_the_live_block_with_no_stored_state(self, temp_home: Path):
        # The block itself says which slot is active, so there is nothing to
        # keep in sync and nothing to go stale.
        _seed_oauth_login()
        s = _switcher()
        s.add_account()
        s.add_provider_account(_bearer_config())

        s.switch_to("2")
        assert s.active_provider_slot() == "2"
        s.switch_to("1")
        assert s.active_provider_slot() is None

    def test_survives_a_hand_edited_block(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        s.switch_to("1")
        provider.apply_block({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "xx"})
        # No longer any slot's config — reported unmanaged, repaired by --force.
        assert s.active_provider_slot() is None
        s.switch_to("1", force=True)
        assert s.active_provider_slot() == "1"


# ---------------------------------------------------------------------------
# Usage display
# ---------------------------------------------------------------------------


class TestUsageDisplay:
    def test_usage_fields_maps_the_sentinel(self):
        assert usage_fields(USAGE_PROVIDER) == ("provider", None)

    def test_sentinel_has_a_human_note(self):
        assert USAGE_PROVIDER in SENTINEL_NOTES
        assert "no quota" in SENTINEL_NOTES[USAGE_PROVIDER]

    def test_collect_short_circuits_without_a_fetch(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_provider_account(_bearer_config())

        def boom(*a, **kw):
            raise AssertionError("a provider account must never be fetched")

        monkeypatch.setattr("claude_swap.oauth.fetch_usage_for_account", boom)
        entries = s._collect_usage_entries(s._build_accounts_info())
        assert entries["1"].sentinel == USAGE_PROVIDER

    def test_list_payload_reports_provider(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        payload = s.list_accounts(json_output=True)
        row = payload["accounts"][0]
        assert row["usageStatus"] == "provider"
        assert row["usage"] is None


# ---------------------------------------------------------------------------
# Autoswitch
# ---------------------------------------------------------------------------


class TestAutoswitch:
    def _engine(self, switcher, **kw):
        from claude_swap.autoswitch import AutoSwitchEngine
        from claude_swap.settings import AutoSwitchSettings

        return AutoSwitchEngine(
            switcher, AutoSwitchSettings(**kw), lambda _e: None
        )

    def test_provider_is_metered(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        engine = self._engine(s)
        assert engine._is_metered("1") is True

    def test_excluded_from_rotation_by_default(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        engine = self._engine(s)
        assert engine._metered_opted_in("1", engine.settings) is False

    def test_its_own_opt_in_gates_it(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        # The API-key opt-in must NOT also enable provider failover.
        api_only = self._engine(s, include_api_key_accounts=True)
        assert api_only._metered_opted_in("1", api_only.settings) is False
        provider_on = self._engine(s, include_provider_accounts=True)
        assert provider_on._metered_opted_in("1", provider_on.settings) is True

    def test_freshening_is_a_noop(self, temp_home: Path):
        # No refresh token exists, so there is nothing to refresh.
        s = _switcher()
        s.add_provider_account(_bearer_config())
        engine = self._engine(s)
        assert engine._freshen_target("1", "bedrock-1@provider.local") == "ok"

    def test_setting_round_trips(self, temp_home: Path):
        from claude_swap.settings import load_settings, set_setting

        root = _switcher().backup_dir
        set_setting(root, "autoswitch.includeProviderAccounts", "true")
        assert load_settings(root).include_provider_accounts is True


# ---------------------------------------------------------------------------
# Session mode
# ---------------------------------------------------------------------------


class TestSessionGuard:
    def test_setup_session_rejects_a_provider_slot(self, temp_home: Path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        with pytest.raises(SessionError, match="third-party provider account"):
            SessionManager(s).setup_session("1", share=False)

    def test_run_rejects_before_any_exec(self, temp_home: Path, monkeypatch):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        monkeypatch.setattr("shutil.which", lambda _n: "/usr/bin/claude")

        def boom(*a, **kw):
            raise AssertionError("must not exec")

        manager = SessionManager(s)
        monkeypatch.setattr(manager, "_exec", boom)
        with pytest.raises(SessionError, match="does not support provider accounts"):
            manager.run("1", [])

    def test_running_an_oauth_slot_under_a_live_provider_warns(
        self, temp_home: Path, capsys
    ):
        # Sharing symlinks the user's settings.json into the profile, and
        # claude applies settings env OVER the process env — so the session runs
        # against the provider whatever credential the profile holds. Verified
        # against claude 2.1.223. Warn: the launch still works, and refusing
        # would strand `cswap run` for anyone defaulting to a provider.
        _write_settings({"env": dict(MANTLE_BLOCK)})
        s = _switcher()
        SessionManager(s)._warn_provider_block_overrides_session()
        out = capsys.readouterr().out
        assert "Amazon Bedrock (Mantle)" in out
        assert "--no-share" in out

    def test_no_warning_without_a_live_provider(self, temp_home: Path, capsys):
        s = _switcher()
        SessionManager(s)._warn_provider_block_overrides_session()
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Export / import
# ---------------------------------------------------------------------------


class TestTransfer:
    def test_export_tags_the_kind_and_keeps_the_config(self, temp_home, tmp_path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        dest = tmp_path / "p.cswap"
        export_accounts(s, str(dest))

        entry = json.loads(dest.read_text(encoding="utf-8"))["accounts"][0]
        assert entry["kind"] == "provider"
        # Every field is load-bearing for activation, so nothing is slimmed —
        # which means a bearer account's secret DOES travel in the export.
        assert entry["credentials"]["env"]["AWS_BEARER_TOKEN_BEDROCK"] == BEARER

    def test_round_trip_preserves_the_config(self, temp_home, tmp_path):
        s = _switcher()
        s.add_provider_account(_bearer_config(name="mantle"))
        dest = tmp_path / "p.cswap"
        export_accounts(s, str(dest))

        s._delete_account_files("1", "mantle-1@provider.local")
        data = s._get_sequence_data()
        data["accounts"] = {}
        data["sequence"] = []
        s._write_json(s.sequence_file, data)

        import_accounts(s, str(dest))
        assert s.account_kind_for("1") == "provider"
        restored = s._provider_config_for("1", "mantle-1@provider.local")
        assert restored.provider == "mantle"
        assert restored.env["AWS_BEARER_TOKEN_BEDROCK"] == BEARER

    def test_imported_config_is_validated(self, temp_home, tmp_path):
        s = _switcher()
        s.add_provider_account(_bearer_config())
        dest = tmp_path / "p.cswap"
        export_accounts(s, str(dest))

        envelope = json.loads(dest.read_text(encoding="utf-8"))
        envelope["accounts"][0]["credentials"]["env"]["TOTALLY_UNRELATED"] = "x"
        dest.write_text(json.dumps(envelope), encoding="utf-8")

        from claude_swap.exceptions import TransferError

        with pytest.raises(TransferError, match="not a provider variable"):
            import_accounts(s, str(dest), force=True)
