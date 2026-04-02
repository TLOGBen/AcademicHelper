"""Gap discovery tools — GapPipeline ABC and find_gaps MCP tool."""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from academic_helper.models.domain import NURSING_TW, DomainProfile
from academic_helper.models.paper import Paper
from academic_helper.tools.search import search_papers

# ---------------------------------------------------------------------------
# GapResult — frozen dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapResult:
    """Immutable result of a gap discovery analysis."""

    gap_type: str
    description: str
    supporting_papers: tuple[Paper, ...]
    confidence: float


# ---------------------------------------------------------------------------
# GapPipeline — ABC with template method run()
# ---------------------------------------------------------------------------


class GapPipeline(ABC):
    """Abstract base for gap discovery pipelines (template method pattern)."""

    gap_type: str  # must be set by each concrete subclass

    def __init__(self, search_fn: Callable) -> None:
        self._search_fn = search_fn

    async def run(self, paper: Paper, domain: DomainProfile) -> tuple[GapResult, ...]:
        """Template method: hypothesize → search → verify."""
        hypothesis = self.hypothesize(paper, domain)
        search_result = self.search(hypothesis, domain)
        if inspect.isawaitable(search_result):
            papers = await search_result
        else:
            papers = search_result
        return self.verify(hypothesis, papers, domain)

    @abstractmethod
    def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
        """Generate a search hypothesis from the given paper and domain."""

    @abstractmethod
    async def search(
        self, hypothesis: str, domain: DomainProfile
    ) -> tuple[Paper, ...]:
        """Search for papers relevant to the hypothesis."""

    @abstractmethod
    def verify(
        self,
        hypothesis: str,
        papers: tuple[Paper, ...],
        domain: DomainProfile,
    ) -> tuple[GapResult, ...]:
        """Verify hypothesis against found papers and return GapResults."""


# ---------------------------------------------------------------------------
# Concrete pipeline subclasses
# ---------------------------------------------------------------------------


class AspectComparison(GapPipeline):
    """Finds papers that outperform specific methodological aspects."""

    gap_type = "aspect_comparison"

    def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
        keywords = " ".join(domain.keywords_hint[:2]) if domain.keywords_hint else domain.field
        return f"{paper.title} methodological comparison {keywords}"

    async def search(
        self, hypothesis: str, domain: DomainProfile
    ) -> tuple[Paper, ...]:
        return await self._search_fn(hypothesis, domain)

    def verify(
        self,
        hypothesis: str,
        papers: tuple[Paper, ...],
        domain: DomainProfile,
    ) -> tuple[GapResult, ...]:
        if not papers:
            return ()
        confidence = min(1.0, len(papers) * 0.2)
        description = f"Papers with stronger methodological aspects found for: {hypothesis}"
        return (
            GapResult(
                gap_type=self.gap_type,
                description=description,
                supporting_papers=papers,
                confidence=confidence,
            ),
        )


class ContextLocalization(GapPipeline):
    """Checks locale applicability of a study."""

    gap_type = "context_localization"

    def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
        return f"{paper.title} {domain.locale} context applicability"

    async def search(
        self, hypothesis: str, domain: DomainProfile
    ) -> tuple[Paper, ...]:
        return await self._search_fn(hypothesis, domain)

    def verify(
        self,
        hypothesis: str,
        papers: tuple[Paper, ...],
        domain: DomainProfile,
    ) -> tuple[GapResult, ...]:
        if not papers:
            return ()
        confidence = min(1.0, len(papers) * 0.25)
        description = f"Context localization gap identified for locale {domain.locale}: {hypothesis}"
        return (
            GapResult(
                gap_type=self.gap_type,
                description=description,
                supporting_papers=papers,
                confidence=confidence,
            ),
        )


class MethodModernization(GapPipeline):
    """Checks whether newer technologies could improve traditional methods."""

    gap_type = "method_modernization"

    def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
        norms = " ".join(domain.methodology_norms[:2]) if domain.methodology_norms else ""
        return f"{paper.title} modern technology improvement {norms}"

    async def search(
        self, hypothesis: str, domain: DomainProfile
    ) -> tuple[Paper, ...]:
        return await self._search_fn(hypothesis, domain)

    def verify(
        self,
        hypothesis: str,
        papers: tuple[Paper, ...],
        domain: DomainProfile,
    ) -> tuple[GapResult, ...]:
        if not papers:
            return ()
        confidence = min(1.0, len(papers) * 0.3)
        description = f"Method modernization opportunity detected: {hypothesis}"
        return (
            GapResult(
                gap_type=self.gap_type,
                description=description,
                supporting_papers=papers,
                confidence=confidence,
            ),
        )


# ---------------------------------------------------------------------------
# Mode mapping — English names and Chinese user-goal phrases
# Maps mode strings to the canonical English class name so patches work.
# ---------------------------------------------------------------------------

_MODE_TO_CLASS_NAME: dict[str, str] = {
    "aspect_comparison": "AspectComparison",
    "這篇論文的某些方法，有沒有其他論文做得更好？": "AspectComparison",
    "context_localization": "ContextLocalization",
    "這篇研究搬到我的情境適用嗎？": "ContextLocalization",
    "method_modernization": "MethodModernization",
    "這篇論文的傳統方法，可以引入新技術改善嗎？": "MethodModernization",
}

import sys as _sys


def _resolve_pipeline_cls(mode: str) -> type[GapPipeline]:
    """Resolve mode string to a pipeline class, respecting any active patches."""
    class_name = _MODE_TO_CLASS_NAME.get(mode)
    if class_name is None:
        raise ValueError(f"Unknown mode {mode!r}. Valid modes: {list(_MODE_TO_CLASS_NAME.keys())}")
    module = _sys.modules[__name__]
    return getattr(module, class_name)

# ---------------------------------------------------------------------------
# Helper: convert a search_papers dict response to Paper objects
# ---------------------------------------------------------------------------


def _dict_to_paper(d: dict) -> Paper:
    authors_raw = d.get("authors") or []
    if isinstance(authors_raw, (list, tuple)):
        authors: tuple[str, ...] = tuple(str(a) for a in authors_raw)
    else:
        authors = (str(authors_raw),)
    return Paper(
        title=d.get("title", ""),
        abstract=d.get("abstract", ""),
        doi=d.get("doi"),
        authors=authors,
        year=d.get("year"),
        source=d.get("source", ""),
        url=d.get("url"),
        citation_count=d.get("citation_count", 0),
    )


def _gap_result_to_dict(result: GapResult) -> dict:
    return {
        "gap_type": result.gap_type,
        "description": result.description,
        "confidence": result.confidence,
        "supporting_papers": [
            {
                "title": p.title,
                "abstract": p.abstract,
                "doi": p.doi,
                "authors": list(p.authors),
                "year": p.year,
                "source": p.source,
                "url": p.url,
                "citation_count": p.citation_count,
            }
            for p in result.supporting_papers
        ],
    }


# ---------------------------------------------------------------------------
# find_gaps MCP tool
# ---------------------------------------------------------------------------


async def find_gaps(
    paper_title: str,
    paper_abstract: str,
    mode: str = "aspect_comparison",
    domain: str = "nursing",
) -> list[dict]:
    """Discover research gaps for a paper using the specified analysis mode.

    Parameters
    ----------
    paper_title:    Title of the paper to analyse.
    paper_abstract: Abstract of the paper to analyse.
    mode:           Analysis mode (English name or Chinese user-goal phrase).
    domain:         Target domain profile name (only 'nursing' supported now).

    Returns
    -------
    A list of gap result dicts, each with gap_type, description, confidence,
    and supporting_papers.
    """
    pipeline_cls = _resolve_pipeline_cls(mode)

    domain_profile = NURSING_TW  # only nursing supported currently

    async def _search_fn(hypothesis: str, _domain: DomainProfile) -> tuple[Paper, ...]:
        raw = await search_papers(query=hypothesis, limit=10)
        return tuple(_dict_to_paper(d) for d in raw)

    paper = Paper(
        title=paper_title,
        abstract=paper_abstract,
        doi=None,
        authors=(),
        year=None,
        source="user_input",
        url=None,
        citation_count=0,
    )

    pipeline = pipeline_cls(search_fn=_search_fn)
    results = await pipeline.run(paper, domain_profile)
    return [_gap_result_to_dict(r) for r in results]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp) -> None:
    """Register find_gaps as an MCP tool on the given FastMCP instance."""
    mcp.tool()(find_gaps)
