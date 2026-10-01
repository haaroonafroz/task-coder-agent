"""Session-level LLM token accounting."""

from __future__ import annotations

import json
from pathlib import Path

from src.agents.llm_stream_events import LLMStreamContext
from src.events import EventEmitter
from src.llm_client import LLMResult
from src.session_usage import record_session_tokens


def test_record_session_tokens_accumulates(tmp_path: Path) -> None:
    meta = tmp_path / "session.json"
    meta.write_text(json.dumps({"session_id": "abc", "status": "running"}), encoding="utf-8")
    first = record_session_tokens(meta, prompt=100, generated=20)
    second = record_session_tokens(meta, prompt=50, generated=10, estimated=True)
    assert first == {"prompt": 100, "generated": 20, "calls": 1, "estimated_calls": 0}
    assert second == {"prompt": 150, "generated": 30, "calls": 2, "estimated_calls": 1}
    saved = json.loads(meta.read_text(encoding="utf-8"))
    assert saved["status"] == "running"
    assert saved["token_usage"] == second


def test_stream_finish_records_session_tokens(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    (tmp_path / "session.json").write_text(json.dumps({"session_id": "s1"}), encoding="utf-8")
    emitter = EventEmitter(events, "s1")
    ctx = LLMStreamContext(emitter=emitter, role="worker")
    result = LLMResult(
        text="ok",
        model_used="gpt4o/gpt-4o",
        prefill_ms=1,
        decode_ms=1,
        total_ms=2,
        tokens_generated=7,
        tokens_prompt=11,
    )
    ctx.finish(result, output_text="ok")
    saved = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert saved["token_usage"]["prompt"] == 11
    assert saved["token_usage"]["generated"] == 7
    assert saved["token_usage"]["calls"] == 1
    types = [json.loads(line)["type"] for line in events.read_text(encoding="utf-8").splitlines()]
    assert types == ["llm.stream.end", "llm.call"]
