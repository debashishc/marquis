from __future__ import annotations

import base64
import sys
import types

import pytest


class _FakeMessage:
    content = "fake-response"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self, calls: list[dict]):
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        return _FakeResponse()


class _FakeChat:
    def __init__(self, calls: list[dict]):
        self.completions = _FakeCompletions(calls)


class _FakeOpenAI:
    calls: list[dict] = []

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _FakeChat(self.calls)


@pytest.fixture()
def fake_openai(monkeypatch):
    _FakeOpenAI.calls = []
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    return _FakeOpenAI


def test_apivlm_text_request_does_not_need_local_vlm_deps(fake_openai) -> None:
    from marquis.common.model_backends import APIVLM

    model = APIVLM(api_base="http://api.test/v1", api_model="qwen-test", api_key="dummy")

    assert model.infer(query="hello") == "fake-response"
    assert fake_openai.calls[-1]["model"] == "qwen-test"
    assert fake_openai.calls[-1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert fake_openai.calls[-1]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_apivlm_video_request_uses_cached_data_uri(tmp_path, fake_openai) -> None:
    from marquis.common.model_backends import APIVLM

    video = tmp_path / "clip.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42fake"
    video.write_bytes(payload)
    expected_uri = "data:video/mp4;base64," + base64.b64encode(payload).decode()
    model = APIVLM(api_base="http://api.test/v1", api_model="qwen-test", api_key="dummy")

    assert model.infer(video_path=str(video), query="describe", fps=3.0) == "fake-response"
    content = fake_openai.calls[-1]["messages"][0]["content"]
    assert content == [
        {"type": "video_url", "video_url": {"url": expected_uri}, "fps": 3.0},
        {"type": "text", "text": "describe"},
    ]
    assert model._video_uri_cache[str(video)] == expected_uri

    video.write_bytes(b"changed")
    assert model.infer(video_path="file://" + str(video), query="again") == "fake-response"
    assert fake_openai.calls[-1]["messages"][0]["content"][0]["video_url"]["url"] == expected_uri
    assert len(model._video_uri_cache) == 1


def test_bullet_api_flags_translate_and_help_mentions_api(capsys) -> None:
    from marquis.article_generation import bullet
    from marquis.article_generation.cli import main

    assert bullet._translate_flags(
        ["--api", "http://api.test/v1", "--api-model", "qwen-test", "data.query_ids=1"]
    ) == ["model.api=http://api.test/v1", "model.api_model=qwen-test", "data.query_ids=1"]
    assert bullet._translate_flags(["--api=http://api.test/v1", "--api-model=qwen-test"]) == [
        "model.api=http://api.test/v1",
        "model.api_model=qwen-test",
    ]

    with pytest.raises(SystemExit, match="--api requires a value"):
        bullet._translate_flags(["--api"])
    with pytest.raises(SystemExit, match="--api-model requires a value"):
        bullet._translate_flags(["--api-model"])

    assert main(["bullet", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--api URL" in help_text
    assert "--api-model NAME" in help_text
