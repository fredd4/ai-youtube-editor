"""Unit tests for the AI client layer (ytedit/ai/*).

Offline tests use ``httpx.MockTransport`` - no network, no keys, no cost.
Tests marked ``live`` hit the real APIs and are skipped unless ``YTEDIT_LIVE=1``
is set; they cost a few cents in total and never touch video generation or
music (both are expensive) unless ``YTEDIT_LIVE_MUSIC=1`` is also set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import httpx
import pytest

from ytedit.ai.elevenlabs import (
    ElevenLabs,
    ElevenLabsError,
    normalize_transcript,
    parse_event_type,
    parse_multipart_mixed,
)
from ytedit.ai.fal import SEEDANCE_USD_PER_SECOND, Fal, FalError, map_status
from ytedit.ai.openrouter import (
    OpenRouter,
    OpenRouterError,
    estimate_cost,
    extract_json,
    find_json_block,
    message_text,
    price_for_model,
    strip_code_fences,
    user_message,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE = os.environ.get("YTEDIT_LIVE") == "1"
live = pytest.mark.skipif(not LIVE, reason="set YTEDIT_LIVE=1 to run live API tests")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def chat_body(content, *, cost=None, extra_message=None, model="test/model"):
    message = {"role": "assistant", "content": content}
    if extra_message:
        message.update(extra_message)
    usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "gen-1", "model": model, "choices": [{"message": message}], "usage": usage}


def make_client(handler, **kwargs) -> OpenRouter:
    transport = httpx.MockTransport(handler)
    return OpenRouter("test-key", client=httpx.Client(transport=transport), **kwargs)


def env_key(name: str) -> str:
    """Read a key from the process env, falling back to the repo .env file."""
    value = os.environ.get(name)
    if value:
        return value
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, val = line.partition("=")
            if sep and key.strip() == name:
                return val.strip()
    return ""


# --------------------------------------------------------------------------- #
# OpenRouter: message text extraction
# --------------------------------------------------------------------------- #


def test_message_text_plain_string():
    assert message_text({"content": "hello"}) == "hello"


def test_message_text_content_none_falls_back_to_reasoning():
    msg = {"content": None, "reasoning": '{"a": 1}'}
    assert message_text(msg) == '{"a": 1}'


def test_message_text_content_none_prefers_parts_over_reasoning():
    msg = {
        "content": None,
        "reasoning": "thinking out loud",
        "parts": [{"type": "thought", "text": "hmm"}, {"type": "text", "text": '{"a": 1}'}],
    }
    assert message_text(msg) == '{"a": 1}'


def test_message_text_list_of_parts():
    msg = {"content": [{"type": "text", "text": "{"}, {"type": "text", "text": '"a": 1}'}]}
    assert json.loads(message_text(msg).replace(" ", "", 1)) == {"a": 1}


def test_message_text_empty_message():
    assert message_text({}) == ""
    assert message_text({"content": ""}) == ""


# --------------------------------------------------------------------------- #
# OpenRouter: JSON extraction
# --------------------------------------------------------------------------- #


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = '```json\n{\n  "dominant": "red",\n  "is_test_pattern": true\n}\n```'
    assert extract_json(text) == {"dominant": "red", "is_test_pattern": True}


def test_extract_json_fenced_without_language():
    assert extract_json("```\n[1, 2, 3]\n```") == [1, 2, 3]


def test_extract_json_unterminated_fence():
    assert extract_json('```json\n{"a": 1}') == {"a": 1}


def test_extract_json_trailing_prose():
    text = 'Sure! Here is the analysis:\n{"clips": [1, 2]}\nLet me know if you need more.'
    assert extract_json(text) == {"clips": [1, 2]}


def test_extract_json_array_with_prose():
    text = "Here you go: [{\"s\": 0.0, \"e\": 1.5}] -- done."
    assert extract_json(text) == [{"s": 0.0, "e": 1.5}]


def test_extract_json_braces_inside_strings():
    text = 'blah {"note": "use {curly} braces \\" here", "n": 2} trailing'
    assert extract_json(text) == {"note": 'use {curly} braces " here', "n": 2}


def test_extract_json_raises_on_garbage():
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json at all")


def test_extract_json_raises_on_empty():
    with pytest.raises(json.JSONDecodeError):
        extract_json("")


def test_find_json_block_skips_unbalanced_opener():
    assert find_json_block('{ oops unbalanced') is None


def test_strip_code_fences_noop_without_fence():
    assert strip_code_fences("  plain text  ") == "plain text"


# --------------------------------------------------------------------------- #
# OpenRouter: pricing
# --------------------------------------------------------------------------- #


def test_price_for_model_batch_is_half():
    assert price_for_model("anthropic/claude-opus-5") == (5.0, 25.0)
    assert price_for_model("anthropic/claude-opus-5:batch") == (2.5, 12.5)
    assert price_for_model("anthropic/claude-opus-5:free") == (0.0, 0.0)
    assert price_for_model("who/knows") is None


def test_estimate_cost():
    # 1M prompt + 1M completion tokens on sonnet-5 = $2 + $10
    assert estimate_cost("anthropic/claude-sonnet-5", 1_000_000, 1_000_000) == pytest.approx(12.0)
    assert estimate_cost("unknown/model", 1000, 1000) == 0.0


# --------------------------------------------------------------------------- #
# OpenRouter: chat over MockTransport
# --------------------------------------------------------------------------- #


def test_chat_sends_usage_include_and_uses_reported_cost():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=chat_body("hi", cost=0.000283))

    costs = []
    client = make_client(handler, cost_callback=lambda **kw: costs.append(kw))
    result = client.chat("anthropic/claude-haiku-4.5", [{"role": "user", "content": "yo"}])

    assert seen["payload"]["usage"] == {"include": True}
    assert seen["headers"]["x-title"] == "ytedit"
    assert result.text == "hi"
    assert result.cost_usd == pytest.approx(0.000283)
    assert result.cost_estimated is False
    assert costs == [
        {
            "service": "openrouter",
            "op": "chat",
            "model": "anthropic/claude-haiku-4.5",
            "units": "100+50 tok",
            "usd": pytest.approx(0.000283),
        }
    ]


def test_chat_estimates_cost_when_usage_cost_missing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_body("hi"))

    client = make_client(handler)
    result = client.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "yo"}])
    # 100 prompt tokens * $2/M + 50 completion * $10/M
    assert result.cost_usd == pytest.approx(100 / 1e6 * 2 + 50 / 1e6 * 10)
    assert result.cost_estimated is True


def test_chat_json_mode_parses_fenced_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=chat_body('```json\n{"ok": true}\n```'))

    client = make_client(handler)
    result = client.chat("m", [{"role": "user", "content": "x"}], json_mode=True)
    assert result.json == {"ok": True}


def test_chat_json_schema_wraps_bare_schema():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        rf = payload["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == {"type": "object"}
        return httpx.Response(200, json=chat_body('{"ok": 1}'))

    client = make_client(handler)
    assert client.chat("m", [], json_schema={"type": "object"}).json == {"ok": 1}


def test_chat_retries_on_429_and_respects_retry_after(monkeypatch):
    slept = []
    monkeypatch.setattr("ytedit.ai.openrouter.time.sleep", slept.append)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": "slow down"})
        return httpx.Response(200, json=chat_body("done", cost=0.0))

    client = make_client(handler)
    assert client.chat("m", []).text == "done"
    assert calls["n"] == 2
    assert slept and slept[0] >= 7.0


def test_chat_raises_on_persistent_500(monkeypatch):
    monkeypatch.setattr("ytedit.ai.openrouter.time.sleep", lambda *_: None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(OpenRouterError) as exc:
        make_client(handler).chat("m", [], retries=2)
    assert exc.value.status == 500


def test_chat_raises_on_error_envelope_with_http_200():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": 402, "message": "no credit"}})

    with pytest.raises(OpenRouterError, match="no credit"):
        make_client(handler).chat("m", [])


def test_ask_json_falls_back_when_response_format_rejected():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append("response_format" in payload)
        if "response_format" in payload:
            return httpx.Response(400, text="response_format not supported by provider")
        return httpx.Response(200, json=chat_body('{"ok": true}'))

    client = make_client(handler)
    assert client.ask_json("m", "sys", "user", '{"ok": bool}') == {"ok": True}
    assert seen == [True, False]


def test_ask_json_retries_once_then_raises(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=chat_body("I am afraid I cannot do that."))

    with pytest.raises(OpenRouterError, match="parseable JSON"):
        make_client(handler).ask_json("m", "sys", "user", "{}")
    assert calls["n"] == 2


def test_ask_json_recovers_from_thinking_mode_content_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chat_body(None, extra_message={"reasoning": 'Final answer:\n{"takes": []}'}),
        )

    assert make_client(handler).ask_json("m", "s", "u", "{}") == {"takes": []}


def test_list_models_is_cached():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"id": "a/b"}]})

    client = make_client(handler)
    client.list_models()
    client.list_models()
    assert calls["n"] == 1
    assert client.model_ids() == {"a/b"}
    client.list_models(refresh=True)
    assert calls["n"] == 2


def test_transcribe_audio_normalizes_words(tmp_path):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake-mp3")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["timestamp_granularities"] == ["word"]
        assert payload["input_audio"]["format"] == "mp3"
        return httpx.Response(
            200,
            json={
                "text": " Dzien dobry.",
                "language": "pl",
                "duration": 1.0,
                "words": [{"word": " Dzien", "start": 0.0, "end": 0.56}],
                "usage": {"seconds": 1, "cost": 0.0000075},
            },
        )

    costs = []
    client = make_client(handler, cost_callback=lambda **kw: costs.append(kw))
    out = client.transcribe_audio(audio, language="pl")
    assert out["words"] == [{"t": " Dzien", "s": 0.0, "e": 0.56}]
    assert out["cost_usd"] == pytest.approx(0.0000075)
    assert out["engine"] == "openrouter/openai/whisper-large-v3"
    assert costs[0]["op"] == "transcribe"


# --------------------------------------------------------------------------- #
# OpenRouter: multimodal message building
# --------------------------------------------------------------------------- #


def test_user_message_plain_text_stays_a_string():
    assert user_message("hello") == {"role": "user", "content": "hello"}


def test_user_message_interleaves_labels_before_images():
    png = b"\x89PNG\r\n\x1a\n" + b"rest"
    msg = user_message("rate these", images=[b"\xff\xd8jpegbytes", png], labels=[0.0, 12.42])
    kinds = [part["type"] for part in msg["content"]]
    assert kinds == ["text", "text", "image_url", "text", "image_url"]
    assert msg["content"][1]["text"] == "Frame 0 (t=0.0s):"
    assert msg["content"][3]["text"] == "Frame 1 (t=12.4s):"
    assert msg["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert msg["content"][4]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msg["content"][2]["image_url"]["detail"] == "low"


def test_user_message_string_labels_used_verbatim():
    msg = user_message("x", images=[b"\xff\xd8a"], labels=["Poster frame:"])
    assert msg["content"][1]["text"] == "Poster frame:"


# --------------------------------------------------------------------------- #
# ElevenLabs
# --------------------------------------------------------------------------- #


def el_client(handler, **kwargs) -> ElevenLabs:
    transport = httpx.MockTransport(handler)
    return ElevenLabs("test-key", client=httpx.Client(transport=transport), **kwargs)


RAW_SCRIBE = {
    "language_code": "pol",
    "language_probability": 0.98,
    "text": "Dzien dobry (laughter) tutaj",
    "audio_duration_secs": 3600.0,
    "words": [
        {"text": "Dzien", "start": 0.32, "end": 0.72, "type": "word", "speaker_id": "speaker_0", "logprob": -0.04},
        {"text": " ", "start": 0.72, "end": 0.74, "type": "spacing"},
        {"text": "dobry", "start": 0.74, "end": 1.1, "type": "word", "speaker_id": "speaker_0", "logprob": -0.1},
        {"text": "(laughter)", "start": 5.1, "end": 6.0, "type": "audio_event"},
        {"text": "tutaj", "start": 6.2, "end": 6.9, "type": "word", "speaker_id": "speaker_1", "logprob": -0.2},
    ],
}


def test_normalize_transcript_matches_project_schema():
    tr = normalize_transcript(RAW_SCRIBE)
    assert tr.language == "pol"
    assert tr.language_probability == 0.98
    assert [w["t"] for w in tr.words] == ["Dzien", "dobry", "tutaj"]
    assert tr.words[0] == {"t": "Dzien", "s": 0.32, "e": 0.72, "p": -0.04, "speaker": "speaker_0"}
    assert tr.events == [{"type": "laughter", "s": 5.1, "e": 6.0}]
    assert tr.speakers == ["speaker_0", "speaker_1"]
    assert tr.engine == "elevenlabs/scribe_v2"
    assert tr.cost_usd == pytest.approx(0.22)  # exactly one hour
    assert set(tr.to_dict()) == {
        "language",
        "language_probability",
        "text",
        "words",
        "events",
        "speakers",
        "engine",
    }


def test_normalize_transcript_without_duration_uses_last_word_end():
    raw = {"words": [{"text": "a", "start": 0.0, "end": 2.5, "type": "word"}]}
    assert normalize_transcript(raw).duration_s == pytest.approx(2.5)


def test_parse_event_type():
    assert parse_event_type("(laughter)") == "laughter"
    assert parse_event_type("(door slam)") == "door_slam"
    # Polish transcripts label events with square brackets, e.g. "[dzwiek]".
    assert parse_event_type("[dzwiek]") == "dzwiek"
    assert parse_event_type("MUSIC") == "music"


def test_transcribe_sends_expected_multipart_fields(tmp_path):
    wav = tmp_path / "c001.wav"
    wav.write_bytes(b"RIFFfake")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode("utf-8", "replace")
        seen["key"] = request.headers["xi-api-key"]
        return httpx.Response(200, json=RAW_SCRIBE)

    costs = []
    tr = el_client(handler, cost_callback=lambda **kw: costs.append(kw)).transcribe(
        wav, language_code="pol", keyterms=["Lizbona", "Alfama"]
    )
    assert seen["url"].endswith("/v1/speech-to-text")
    assert seen["key"] == "test-key"
    # keyterms must be repeated form fields, not one JSON array string
    assert seen["body"].count('name="keyterms"') == 2
    assert "[" not in seen["body"].split('name="keyterms"')[1][:40]
    for expected in (
        'name="model_id"',
        "scribe_v2",
        'name="timestamps_granularity"',
        'name="no_verbatim"',
        'name="keyterms"',
        "Lizbona",
        "Alfama",
        'name="file"; filename="c001.wav"',
    ):
        assert expected in seen["body"], expected
    assert "false" in seen["body"]  # no_verbatim stays off
    assert tr.text.startswith("Dzien")
    assert costs[0] == {
        "service": "elevenlabs",
        "op": "stt",
        "model": "scribe_v2",
        "units": "3600.0s",
        "usd": pytest.approx(0.22),
    }


def test_compose_music_cost_callback_and_request_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        seen["url"] = str(request.url)
        seen["accept"] = request.headers["accept"]
        return httpx.Response(
            200, content=b"ID3fake-mp3", headers={"song-id": "song-abc", "content-type": "audio/mpeg"}
        )

    costs = []
    client = el_client(handler, cost_callback=lambda **kw: costs.append(kw))
    result = client.compose_music("warm acoustic travel bed", 90_000)

    assert seen["payload"]["model_id"] == "music_v2"  # never rely on the v1 default
    assert seen["payload"]["force_instrumental"] is True
    assert seen["payload"]["generation_mode"] == "loop"
    assert seen["payload"]["music_length_ms"] == 90_000
    assert "output_format=mp3_44100_192" in seen["url"]
    assert seen["accept"] == "audio/mpeg"
    assert result.audio == b"ID3fake-mp3"
    assert result.song_id == "song-abc"
    # 90 s at $0.15/min
    assert costs == [
        {
            "service": "elevenlabs",
            "op": "music",
            "model": "music_v2",
            "units": "90.0s",
            "usd": pytest.approx(0.225),
        }
    ]
    assert result.meta["cost_usd"] == pytest.approx(0.225)


def test_compose_music_rejects_out_of_range_length():
    with pytest.raises(ValueError):
        el_client(lambda r: httpx.Response(200)).compose_music("x", 1000)


def test_compose_music_writes_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"mp3", headers={"song-id": "s1"})

    out = el_client(handler).compose_music("x", 3000).write(tmp_path / "beds" / "a.mp3")
    assert out.read_bytes() == b"mp3"


def test_tts_cost_uses_per_model_price():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model_id"] == "eleven_flash_v2_5"
        assert payload["language_code"] == "pl"
        return httpx.Response(200, content=b"audio")

    costs = []
    client = el_client(handler, cost_callback=lambda **kw: costs.append(kw))
    client.tts("x" * 1000, "voice1", model_id="eleven_flash_v2_5")
    assert costs[0]["usd"] == pytest.approx(0.05)


def test_tts_falls_back_when_output_format_is_plan_gated():
    """payg accounts get 403 output_format_not_allowed for mp3_44100_192."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "mp3_44100_192" in str(request.url):
            return httpx.Response(
                403,
                json={
                    "detail": {
                        "type": "authorization_error",
                        "status": "output_format_not_allowed",
                        "message": "Output format 'mp3_44100_192' is only available on the Creator tier and above.",
                    }
                },
            )
        return httpx.Response(200, content=b"audio")

    assert el_client(handler).tts("czesc", "v1") == b"audio"
    assert len(seen) == 2
    assert "mp3_44100_128" in seen[1]


def test_tts_does_not_swallow_other_403s():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": {"status": "voice_not_allowed"}})

    with pytest.raises(ElevenLabsError):
        el_client(handler).tts("czesc", "v1")


def test_error_body_is_truncated():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, text="e" * 5000)

    with pytest.raises(ElevenLabsError) as exc:
        el_client(handler).subscription()
    assert exc.value.status == 422
    assert len(exc.value.body) == 500


def test_elevenlabs_retries_on_503(monkeypatch):
    monkeypatch.setattr("ytedit.ai.elevenlabs.time.sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"tier": "payg"})

    assert el_client(handler).subscription() == {"tier": "payg"}
    assert calls["n"] == 3


def test_parse_multipart_mixed():
    body = (
        b"--BOUND\r\nContent-Type: application/json\r\n\r\n{\"song_id\": \"x\"}\r\n"
        b"--BOUND\r\nContent-Type: audio/mpeg\r\n\r\nID3audio\r\n--BOUND--\r\n"
    )
    parts = parse_multipart_mixed("multipart/mixed; boundary=BOUND", body)
    assert len(parts) == 2
    assert json.loads(parts[0][1]) == {"song_id": "x"}
    assert parts[1][1] == b"ID3audio"


def test_compose_music_detailed_splits_multipart():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            b"--B\r\nContent-Type: application/json\r\n\r\n{\"song_id\": \"sid\"}\r\n"
            b"--B\r\nContent-Type: audio/mpeg\r\n\r\nAUDIO\r\n--B--\r\n"
        )
        return httpx.Response(
            200, content=body, headers={"content-type": "multipart/mixed; boundary=B"}
        )

    plan = {
        "positive_global_styles": ["warm"],
        "negative_global_styles": ["drums"],
        "sections": [{"section_name": "intro", "duration_ms": 30_000}],
    }
    result = el_client(handler).compose_music_detailed(plan)
    assert result.audio == b"AUDIO"
    assert result.song_id == "sid"
    assert result.meta["cost_usd"] == pytest.approx(0.075)


# --------------------------------------------------------------------------- #
# fal
# --------------------------------------------------------------------------- #


# Names mirror the fal_client status classes exactly - map_status() keys off
# the class name, so a renamed stub would silently pass a broken mapping.
class Queued:
    pass


class InProgress:
    pass


class Completed:
    pass


class Failed:
    pass


def test_map_status_class_names():
    assert map_status(Queued()) == "queued"
    assert map_status(InProgress()) == "in_progress"
    assert map_status(Completed()) == "completed"
    assert map_status(Failed()) == "failed"


def test_map_status_accepts_strings_and_unknowns():
    assert map_status("Completed") == "completed"
    assert map_status("SomethingNew") == "something_new"


def test_fal_status_uses_class_name(monkeypatch):
    fal = Fal("key")
    calls = []

    class FakeSdk:
        @staticmethod
        def status(app, request_id, with_logs=False):
            calls.append((app, request_id, with_logs))
            return InProgress()

    fal._sdk = FakeSdk
    assert fal.status("app/x", "rid") == "in_progress"
    assert calls == [("app/x", "rid", False)]


def test_fal_sets_env_key_on_lazy_import(monkeypatch):
    # A dummy module stands in for fal_client: the real SDK builds its global
    # HTTP client (and caches credentials) at import time, so importing it here
    # with a fake key would poison every later call in the session.
    monkeypatch.setitem(sys.modules, "fal_client", types.ModuleType("fal_client"))
    # setenv (not delenv) so monkeypatch records an undo entry: without it the
    # fake key would leak into the rest of the session and break live tests.
    monkeypatch.setenv("FAL_KEY", "placeholder")
    fal = Fal("secret-key")
    assert os.environ["FAL_KEY"] == "placeholder"  # nothing until the SDK is needed
    assert fal.sdk is sys.modules["fal_client"]
    assert os.environ["FAL_KEY"] == "secret-key"


def test_seedance_requires_confirm():
    fal = Fal("key")
    with pytest.raises(FalError, match="confirm=True"):
        fal.seedance_i2v("frame.jpg", "pan across the bay", duration=5)


def test_seedance_price_table_matches_research():
    assert SEEDANCE_USD_PER_SECOND["720p"] == 0.47
    assert 5 * SEEDANCE_USD_PER_SECOND["720p"] == pytest.approx(2.35)


def test_fal_wait_raises_on_failed(monkeypatch):
    fal = Fal("key")

    class FakeSdk:
        @staticmethod
        def status(app, request_id, with_logs=False):
            return Failed()

    fal._sdk = FakeSdk
    with pytest.raises(FalError, match="failed"):
        fal.wait("app", "rid", poll_s=0)


def test_fal_thumbnail_edit_builds_arguments(monkeypatch, tmp_path):
    src = tmp_path / "face.jpg"
    src.write_bytes(b"\xff\xd8jpeg")
    fal = Fal("key")
    captured = {}

    monkeypatch.setattr(Fal, "upload", lambda self, p: f"https://fal.media/{Path(p).name}")
    def fake_download(self, url, dest):
        Path(dest).write_bytes(b"img")
        return Path(dest)

    monkeypatch.setattr(Fal, "download", fake_download)

    def fake_run(self, app, arguments, with_logs=True):
        captured["app"] = app
        captured["arguments"] = arguments
        return {"images": [{"url": "https://fal.media/out0.jpg"}, {"url": "https://fal.media/out1.jpg"}]}

    monkeypatch.setattr(Fal, "run", fake_run)
    costs = []
    fal.cost_callback = lambda **kw: costs.append(kw)

    out = fal.thumbnail_edit("bold title", [src], n=4, out_dir=tmp_path / "thumbs")
    assert captured["app"] == "fal-ai/nano-banana-pro/edit"
    assert captured["arguments"] == {
        "prompt": "bold title",
        "image_urls": ["https://fal.media/face.jpg"],
        "num_images": 4,
        "aspect_ratio": "16:9",
        "resolution": "2K",
        "output_format": "jpeg",
    }
    assert [p.name for p in out] == ["thumb_00.jpg", "thumb_01.jpg"]
    assert costs[0]["usd"] == pytest.approx(0.60)


def test_fal_thumbnail_edit_4k_doubles_cost(monkeypatch, tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"\xff\xd8")
    fal = Fal("key")
    monkeypatch.setattr(Fal, "upload", lambda self, p: "u")
    monkeypatch.setattr(Fal, "download", lambda self, url, dest: Path(dest))
    monkeypatch.setattr(Fal, "run", lambda self, a, args, with_logs=True: {"images": [{"url": "u"}]})
    costs = []
    fal.cost_callback = lambda **kw: costs.append(kw)
    fal.thumbnail_edit("p", [src], n=2, resolution="4K", out_dir=tmp_path)
    assert costs[0]["usd"] == pytest.approx(0.60)


def test_fal_no_images_raises(monkeypatch, tmp_path):
    src = tmp_path / "a.jpg"
    src.write_bytes(b"\xff\xd8")
    fal = Fal("key")
    monkeypatch.setattr(Fal, "upload", lambda self, p: "u")
    monkeypatch.setattr(Fal, "run", lambda self, a, args, with_logs=True: {"images": []})
    with pytest.raises(FalError, match="no images"):
        fal.thumbnail_edit("p", [src], out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Live tests (YTEDIT_LIVE=1)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def test_jpg(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "frame.jpg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=320x180:duration=1", "-frames:v", "1", str(path)],
        check=True,
    )
    return path


@live
def test_live_openrouter_vision(test_jpg):
    """Cheap vision round trip: ~$0.0003 on claude-haiku-4.5."""
    client = OpenRouter(env_key("OPENROUTER_API_KEY"))
    out = client.ask_json(
        "anthropic/claude-haiku-4.5",
        "You are a terse frame QC assistant.",
        "Describe the frame.",
        '{"dominant_color": string, "is_test_pattern": boolean}',
        images=[test_jpg],
        labels=[0.0],
        max_tokens=200,
    )
    assert isinstance(out, dict)
    assert out["is_test_pattern"] is True


@live
def test_live_openrouter_transcribe(tmp_path):
    """Whisper fallback on a 1 s tone; returns something with word timestamps."""
    audio = tmp_path / "tone.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", "-ar", "16000", "-ac", "1", str(audio)],
        check=True,
    )
    out = OpenRouter(env_key("OPENROUTER_API_KEY")).transcribe_audio(audio, language="pl")
    assert "text" in out and isinstance(out["words"], list)
    assert out["cost_usd"] < 0.001


@live
def test_live_elevenlabs_subscription():
    sub = ElevenLabs(env_key("ELEVENLABS_API_KEY")).subscription()
    assert sub["status"] == "active"
    assert sub["character_count"] <= sub["character_limit"]


@live
def test_live_elevenlabs_tts_stt_roundtrip(tmp_path):
    """Polish TTS -> Scribe v2 STT: verifies words, order and timestamps.

    Cost: ~$0.005 TTS (eleven_flash_v2_5, ~100 chars) + ~$0.0004 STT.
    """
    client = ElevenLabs(env_key("ELEVENLABS_API_KEY"))
    voice_id = next(v["voice_id"] for v in client.list_voices())
    text = "Dzien dobry, jestem w Lizbonie i wlasnie wsiadam do slynnego tramwaju numer dwadziescia osiem."

    # mp3_44100_192 is Creator-tier only; the client degrades automatically.
    audio = client.tts(text, voice_id, model_id="eleven_flash_v2_5", language_code="pl")
    mp3 = tmp_path / "pl.mp3"
    mp3.write_bytes(audio)
    assert mp3.stat().st_size > 5000

    tr = client.transcribe(mp3, language_code="pol", keyterms=["Lizbona"])
    assert tr.language == "pol"
    assert tr.words, "scribe returned no words"
    assert "lizbon" in tr.text.lower()
    # timestamps must be monotonic and inside the clip
    starts = [w["s"] for w in tr.words]
    assert starts == sorted(starts)
    assert tr.words[-1]["e"] <= tr.duration_s + 0.5
    assert all(w["p"] is not None for w in tr.words)


@live
def test_live_fal_upload(test_jpg):
    """Uploads are free; this only proves the key and SDK wiring work."""
    url = Fal(env_key("FAL_KEY")).upload(test_jpg)
    assert url.startswith("https://")


@pytest.mark.skipif(
    os.environ.get("YTEDIT_LIVE_MUSIC") != "1",
    reason="set YTEDIT_LIVE_MUSIC=1 to spend ~$0.01 on a 3 s music generation",
)
@live
def test_live_elevenlabs_music_shortest(tmp_path):
    client = ElevenLabs(env_key("ELEVENLABS_API_KEY"))
    result = client.compose_music(
        "warm nostalgic acoustic travel bed, fingerpicked guitar, 80 BPM. Instrumental.",
        3000,
        generation_mode="loop",
    )
    assert result.audio[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    assert result.meta["cost_usd"] == pytest.approx(0.0075)
