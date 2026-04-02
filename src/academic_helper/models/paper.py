"""Paper dataclass — standardized representation of an academic paper."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    """Immutable representation of an academic paper."""

    title: str
    abstract: str
    doi: str | None
    authors: tuple[str, ...]
    year: int | None
    source: str
    url: str | None
    citation_count: int = 0
