"""SourceResult — fault isolation wrapper for API connector results."""

from dataclasses import dataclass

from academic_helper.models.paper import Paper


@dataclass(frozen=True)
class SourceResult:
    """Immutable result from an academic source search, encapsulating success or failure."""

    source_name: str
    papers: tuple[Paper, ...] = ()
    error: str | None = None
    elapsed_ms: int = 0
