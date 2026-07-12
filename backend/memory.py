"""
memory.py — Lightweight conversation-memory store.

Manages per-session chat history so the LLM has multi-turn context.
Designed to be swapped out for Redis / a database later without touching
the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ─── Types ───────────────────────────────────────────────────
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """A single message in a conversation turn.

    Attributes
    ----------
    role : Role
        One of ``system``, ``user``, ``assistant``, ``tool``.
    content : str | None
        The message body.  May be ``None`` for assistant messages that
        only contain tool calls.
    tool_call_id : str | None
        Required for ``tool`` role messages — the ID of the tool call
        this message is responding to.
    name : str | None
        Optional function name for ``tool`` role messages.
    tool_calls : list[dict] | None
        Present on ``assistant`` messages when the LLM requests tool
        invocations.  Stored as-is from the Groq API response.
    """

    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class ConversationMemory:
    """In-memory store for a single conversation session.

    Attributes
    ----------
    messages : list[Message]
        Ordered list of messages exchanged so far.
    max_turns : int
        Maximum number of *user+assistant* turn pairs to retain
        (the system message is always kept).
    """

    messages: list[Message] = field(default_factory=list)
    max_turns: int = 10

    # ── Public API ───────────────────────────────────────────

    def add_message(
        self,
        role: Role,
        content: str | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a message and enforce the turn limit.

        Parameters
        ----------
        role : Role
            One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
        content : str | None
            The message body.
        tool_call_id : str | None
            ID of the tool call (for ``tool`` role messages).
        name : str | None
            Function name (for ``tool`` role messages).
        tool_calls : list[dict] | None
            Tool call requests (for ``assistant`` role messages).
        """
        self.messages.append(
            Message(
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                name=name,
                tool_calls=tool_calls,
            )
        )
        self._trim()

    def get_messages(self) -> list[dict[str, Any]]:
        """Return all messages as a list of dicts (Groq API format).

        Filters out ``None`` values so the payload is clean for the API.
        Preserves ``tool_calls`` on assistant messages for multi-turn
        tool calling to work correctly.
        """
        result: list[dict[str, Any]] = []
        for msg in self.messages:
            entry: dict[str, Any] = {"role": msg.role}

            # Content may be None for tool-calling assistant messages
            if msg.content is not None:
                entry["content"] = msg.content

            if msg.tool_call_id is not None:
                entry["tool_call_id"] = msg.tool_call_id

            if msg.name is not None:
                entry["name"] = msg.name

            if msg.tool_calls is not None:
                entry["tool_calls"] = msg.tool_calls

            result.append(entry)
        return result

    def clear(self) -> None:
        """Reset the conversation history."""
        self.messages.clear()

    # ── Internal ─────────────────────────────────────────────

    def _trim(self) -> None:
        """Keep only the system message + the last *max_turns* pairs."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system = [m for m in self.messages if m.role != "system"]

        # Enforce turn limits (user + assistant message pairs)
        max_items = self.max_turns * 4  # Allow extra slot space for intermediate tool calls
        if len(non_system) > max_items:
            non_system = non_system[-max_items:]

        # Prune older tool outputs to save context tokens
        last_user_idx = -1
        for idx, msg in enumerate(non_system):
            if msg.role == "user":
                last_user_idx = idx

        for idx, msg in enumerate(non_system):
            if idx < last_user_idx and msg.role == "tool":
                msg.content = "Older tool output truncated."

        self.messages = system_msgs + non_system


# ─── Session registry ───────────────────────────────────────
_sessions: dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    """Return (or create) the ``ConversationMemory`` for *session_id*."""
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def delete_memory(session_id: str) -> None:
    """Remove a session's memory entirely."""
    _sessions.pop(session_id, None)
