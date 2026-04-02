"""Committee review models — ReviewContext for pre-fetched evaluation context."""

from __future__ import annotations

import types
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewContext:
    """Immutable pre-fetched context for a thesis committee evaluation."""

    papers: tuple[dict, ...]
    committee_members: tuple[dict, ...]
    rubric: dict[str, tuple[str, ...]]
    domain: str
    concern: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rubric", types.MappingProxyType(self.rubric))

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict with tuples converted to lists."""
        return {
            "papers": list(self.papers),
            "committee_members": list(self.committee_members),
            "rubric": {k: list(v) for k, v in self.rubric.items()},
            "domain": self.domain,
            "concern": self.concern,
        }
