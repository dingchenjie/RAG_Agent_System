"""共享黑板：多 Agent 之间的显式协作面。

不用"子 Agent 直接读别人的上下文"，因为那既不可控也不可观测；
用带 owner、version 的黑板，每次写入都产生事件，冲突可见、可回放。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Artifact:
    key: str
    value: str
    owner: str
    version: int = 1
    created_at: float = field(default_factory=time.time)
    history: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value, "owner": self.owner,
                "version": self.version, "created_at": self.created_at,
                "history": self.history}


class Blackboard:
    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def put(self, key: str, value: str, *, owner: str) -> Artifact:
        existing = self._artifacts.get(key)
        if existing is None:
            artifact = Artifact(key=key, value=value, owner=owner)
            self._artifacts[key] = artifact
            return artifact
        existing.history.append(existing.value)
        existing.value = value
        existing.owner = owner
        existing.version += 1
        return existing

    def get(self, key: str) -> Artifact | None:
        return self._artifacts.get(key)

    def keys(self) -> list[str]:
        return list(self._artifacts)

    def snapshot(self) -> dict[str, dict]:
        return {key: artifact.to_dict() for key, artifact in self._artifacts.items()}

    def render(self) -> str:
        if not self._artifacts:
            return ""
        return "\n\n".join(
            f"### {key}（来自 {artifact.owner}）\n{artifact.value}"
            for key, artifact in self._artifacts.items()
        )
