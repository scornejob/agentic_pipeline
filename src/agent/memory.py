"""
Conversation memory — keeps track of the full message history for a session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.provider import Message


@dataclass
class Memory:
    """Simple rolling message buffer."""
    max_messages: int = 100
    _messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))
        # Keep only the last N messages (but always keep the system message)
        if len(self._messages) > self.max_messages:
            system_msgs = [m for m in self._messages if m.role == "system"]
            rest = [m for m in self._messages if m.role != "system"][-self.max_messages + len(system_msgs):]
            self._messages = system_msgs + rest

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def clear_except_system(self) -> None:
        self._messages = [m for m in self._messages if m.role == "system"]

    def set_system(self, content: str) -> None:
        self._messages = [m for m in self._messages if m.role != "system"]
        self._messages.insert(0, Message(role="system", content=content))
