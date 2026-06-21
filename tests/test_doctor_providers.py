"""C24 — Risk Doctor provider selection (ollama / anthropic / perplexity) + the read-only /read reader.

No real API keys/servers needed: SDKs are faked via sys.modules so make_client()/_completion_text()
dispatch is verified offline. The repo reader is checked for read-only + path-traversal safety. The
DEFAULT path (ollama, OpenAI-compat) is unchanged, so the existing test_risk_doctor.py stays green.
"""

import sys
import types

import pytest

from barbershop import config, risk_doctor as rd


@pytest.fixture
def restore_provider():
    orig = config.DOCTOR_PROVIDER
    yield
    config.DOCTOR_PROVIDER = orig


def _fake_openai(record):
    mod = types.ModuleType("openai")
    class OpenAI:
        def __init__(self, **kw): record.update(kw); record["sdk"] = "openai"
    mod.OpenAI = OpenAI
    return mod


def _fake_anthropic(record):
    mod = types.ModuleType("anthropic")
    class Anthropic:
        def __init__(self, **kw): record.update(kw); record["sdk"] = "anthropic"
    mod.Anthropic = Anthropic
    return mod


# ───────────────────────────── config knobs ─────────────────────────────
def test_provider_config_defaults_to_ollama():
    assert config.DOCTOR_PROVIDER in ("ollama", "anthropic", "perplexity")
    assert config.DOCTOR_PERPLEXITY_BASE == "https://api.perplexity.ai"
    assert hasattr(config, "DOCTOR_ANTHROPIC_MODEL") and hasattr(config, "DOCTOR_PERPLEXITY_MODEL")


# ───────────────────────────── make_client dispatch ─────────────────────────────
def test_make_client_perplexity_uses_openai_compat(monkeypatch, restore_provider):
    rec = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(rec))
    config.DOCTOR_PROVIDER = "perplexity"
    rd.make_client()
    assert rec["sdk"] == "openai" and rec["base_url"] == config.DOCTOR_PERPLEXITY_BASE
    assert rec["api_key"] == config.DOCTOR_PERPLEXITY_KEY


def test_make_client_anthropic_uses_anthropic_sdk(monkeypatch, restore_provider):
    rec = {}
    monkeypatch.setitem(sys.modules, "anthropic", _fake_anthropic(rec))
    config.DOCTOR_PROVIDER = "anthropic"
    rd.make_client()
    assert rec["sdk"] == "anthropic"


def test_make_client_default_is_ollama_endpoint(monkeypatch, restore_provider):
    rec = {}
    monkeypatch.setitem(sys.modules, "openai", _fake_openai(rec))
    config.DOCTOR_PROVIDER = "ollama"
    rd.make_client()
    assert rec["base_url"] == config.DOCTOR_API_BASE


# ───────────────────────────── _completion_text dispatch ─────────────────────────────
class _FakeOpenAIClient:
    def __init__(self): self.chat = self
    @property
    def completions(self): return self
    def create(self, **kw):
        self.kw = kw
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="OPENAI-COMPAT REPLY"))])


class _FakeAnthropicClient:
    def __init__(self): self.messages = self
    def create(self, **kw):
        self.kw = kw
        return types.SimpleNamespace(content=[types.SimpleNamespace(text="ANTHROPIC REPLY")])


def test_completion_text_openai_compat(restore_provider):
    config.DOCTOR_PROVIDER = "ollama"
    assert rd._completion_text(_FakeOpenAIClient(), {"system": "s", "user": "u"}) == "OPENAI-COMPAT REPLY"


def test_completion_text_anthropic_uses_messages_api(restore_provider):
    config.DOCTOR_PROVIDER = "anthropic"
    cli = _FakeAnthropicClient()
    assert rd._completion_text(cli, {"system": "SYS", "user": "U"}) == "ANTHROPIC REPLY"
    assert cli.kw["system"] == "SYS" and cli.kw["model"] == config.DOCTOR_ANTHROPIC_MODEL


# ───────────────────────────── repo_file_loader (read-only + safe) ─────────────────────────────
def test_repo_reader_reads_a_real_file():
    out = rd.repo_file_loader("requirements.txt")
    assert "numpy" in out and "[/read denied]" not in out


def test_repo_reader_blocks_path_traversal():
    assert "[/read denied]" in rd.repo_file_loader("../../../etc/passwd")
    assert "/read denied" in rd.repo_file_loader("/etc/passwd")


def test_repo_reader_missing_file_message():
    assert "file not found" in rd.repo_file_loader("quantra/does_not_exist.py")


# ───────────────────────────── /read command in ask() ─────────────────────────────
def test_ask_read_command_returns_file_without_llm():
    # no client + no manual needed: /read is intercepted before the LLM + the context rules
    resp = rd.ask("/read requirements.txt", screen_state={}, log=False)
    assert resp["read_only"] is True and "numpy" in resp["text"] and resp["offline"] is False
