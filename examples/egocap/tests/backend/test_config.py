import pytest

from capture_server import protocol as p
from egocap_web.config import PluginConfigError, SourceConfigError, _parse_plugins, _parse_sources, config_from_env


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_unset_or_blank_plugins_selects_none(raw):
    assert _parse_plugins(raw) == ()


@pytest.mark.parametrize("raw", ["manus", "MANUS", "  Manus  "])
def test_manus_is_accepted_in_any_case_with_surrounding_whitespace(raw):
    assert _parse_plugins(raw) == (p.PluginType.MANUS,)


def test_unknown_plugin_name_fails_naming_the_value_and_the_valid_names():
    with pytest.raises(PluginConfigError, match=r"'oglo' is not a known plugin.*\['manus'\]"):
        _parse_plugins("oglo")


def test_duplicate_plugin_is_rejected():
    with pytest.raises(PluginConfigError, match="more than once"):
        _parse_plugins("manus, manus")


@pytest.mark.parametrize("raw", ["manus,", ",manus", "manus,,manus"])
def test_empty_entry_is_rejected_rather_than_ignored(raw):
    with pytest.raises(PluginConfigError, match="empty entry"):
        _parse_plugins(raw)


def test_every_protocol_plugin_type_is_configurable_by_its_lowercase_name():
    for member in p.PluginType:
        assert _parse_plugins(member.name.lower()) == (member,)


def test_config_from_env_reads_plugins(monkeypatch):
    monkeypatch.delenv("EGOCAP_CAPTURE_PLUGINS", raising=False)
    assert config_from_env().plugins == ()
    monkeypatch.setenv("EGOCAP_CAPTURE_PLUGINS", "manus")
    assert config_from_env().plugins == (p.PluginType.MANUS,)


def test_config_from_env_fails_at_startup_on_a_bad_plugin_name(monkeypatch):
    monkeypatch.setenv("EGOCAP_CAPTURE_PLUGINS", "gloves")
    with pytest.raises(PluginConfigError):
        config_from_env()


@pytest.mark.parametrize("raw", [None, "", "  "])
def test_unset_or_blank_sources_keeps_the_default(raw):
    assert _parse_sources(raw) is None


def test_none_selects_a_cameras_only_session():
    assert _parse_sources(" None ") == ()


def test_sources_are_accepted_in_any_case_with_surrounding_whitespace():
    assert _parse_sources(" Head , hands") == (p.SourceType.HEAD, p.SourceType.HANDS)


@pytest.mark.parametrize("raw", ["gloves", "head,", "head,head"])
def test_a_malformed_sources_value_is_rejected(raw):
    with pytest.raises(SourceConfigError):
        _parse_sources(raw)


def test_config_from_env_reads_sources(monkeypatch):
    monkeypatch.delenv("EGOCAP_CAPTURE_SOURCES", raising=False)
    default = config_from_env().sources
    assert default == (p.SourceType.HEAD, p.SourceType.CONTROLLERS, p.SourceType.HANDS)
    monkeypatch.setenv("EGOCAP_CAPTURE_SOURCES", "none")
    assert config_from_env().sources == ()
