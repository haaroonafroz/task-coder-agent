"""SSE helpers for streaming LLM generation to the chat UI."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.events import EventEmitter


@dataclass
class LLMStreamContext:
    """Optional emitter metadata attached to one ``call_llm`` invocation."""

    emitter: "EventEmitter"
    role: str
    milestone_id: str = ""
    phase: str = ""
    output_kind: str = "text"
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def start(self, *, model_used: str, thinking_level: Optional[str]) -> None:
        payload: dict[str, Any] = {
            "call_id": self.call_id,
            "role": self.role,
            "model_used": model_used,
            "thinking_level": thinking_level,
            "output_kind": self.output_kind,
        }
        if self.milestone_id:
            payload["milestone_id"] = self.milestone_id
        if self.phase:
            payload["phase"] = self.phase
        self.emitter.emit("llm.stream.start", **payload)

    def delta(self, channel: str, text: str) -> None:
        if not text:
            return
        self.emitter.emit(
            "llm.stream.delta",
            call_id=self.call_id,
            role=self.role,
            channel=channel,
            text=text,
        )

    def finish(
        self,
        result: Any,
        *,
        thinking_text: str = "",
        output_text: str = "",
    ) -> None:
        preview_limit = 12_000
        thinking_preview = thinking_text[:preview_limit] if thinking_text else ""
        output_preview = output_text[:preview_limit] if output_text else ""

        shared: dict[str, Any] = {
            "call_id": self.call_id,
            "role": self.role,
            "model_used": result.model_used,
            "tokens_prompt": result.tokens_prompt,
            "tokens_generated": result.tokens_generated,
            "prefill_ms": result.prefill_ms,
            "decode_ms": result.decode_ms,
            "total_ms": result.total_ms,
            "thinking_level": result.thinking_level,
            "fallback_used": result.fallback_used,
            "output_kind": self.output_kind,
            "thinking_chars": len(thinking_text),
            "output_chars": len(output_text),
        }
        if thinking_preview:
            shared["thinking_preview"] = thinking_preview
        if output_preview:
            shared["output_preview"] = output_preview
        if self.milestone_id:
            shared["milestone_id"] = self.milestone_id
        if self.phase:
            shared["phase"] = self.phase

        self.emitter.emit("llm.stream.end", **shared)
        self.emitter.emit("llm.call", **shared)


def stream_context_for(
    emitter: Optional["EventEmitter"],
    role: str,
    *,
    milestone_id: str = "",
    phase: str = "",
    output_kind: str = "text",
) -> Optional[LLMStreamContext]:
    if emitter is None:
        return None
    return LLMStreamContext(
        emitter=emitter,
        role=role,
        milestone_id=milestone_id,
        phase=phase,
        output_kind=output_kind,
    )
