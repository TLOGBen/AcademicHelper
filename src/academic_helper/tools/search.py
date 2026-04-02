"""Search MCP tools — search_papers, deep_search, expand_topics."""

from __future__ import annotations

from academic_helper.models.paper import Paper
from academic_helper.sources.registry import get_all_sources
from academic_helper.sources.search import search_all

# ---------------------------------------------------------------------------
# Topic expansion templates — deterministic, no network calls required
# ---------------------------------------------------------------------------

_TOPIC_EXPANSIONS: dict[str, list[str]] = {}

_EXPANSION_SUFFIXES = [
    "systematic review",
    "meta-analysis",
    "clinical outcomes",
    "evidence-based practice",
    "recent advances",
]


def _generate_related_queries(seed: str, limit: int) -> list[str]:
    """Return deterministic related queries derived from the seed topic."""
    candidates = [f"{seed} {suffix}" for suffix in _EXPANSION_SUFFIXES]
    seen: set[str] = set()
    unique: list[str] = []
    for q in candidates:
        if q not in seen:
            seen.add(q)
            unique.append(q)
        if len(unique) >= limit:
            break
    return unique


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers by DOI, keeping the one with the highest citation_count.

    Papers with doi=None pass through unchanged (no dedup applied).
    """
    best: dict[str, Paper] = {}
    no_doi: list[Paper] = []

    for paper in papers:
        if paper.doi is None:
            no_doi.append(paper)
            continue
        existing = best.get(paper.doi)
        if existing is None or paper.citation_count > existing.citation_count:
            best = {**best, paper.doi: paper}

    return list(best.values()) + no_doi


def _paper_to_dict(paper: Paper) -> dict:
    return {
        "title": paper.title,
        "abstract": paper.abstract,
        "doi": paper.doi,
        "authors": list(paper.authors),
        "year": paper.year,
        "source": paper.source,
        "url": paper.url,
        "citation_count": paper.citation_count,
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def search_papers(query: str, limit: int = 10) -> list[dict]:
    """Search across all sources, deduplicate by DOI, and return paper dicts."""
    sources = get_all_sources()
    results = await search_all(sources, query, limit)

    all_papers: list[Paper] = []
    for result in results:
        if result.error is None:
            all_papers.extend(result.papers)

    deduped = _deduplicate(all_papers)
    return [_paper_to_dict(p) for p in deduped]


async def deep_search(paper_doi: str, depth: int = 1) -> list[dict]:
    """Traverse the citation network starting from paper_doi.

    The seed paper (paper_doi) is excluded from the returned results.
    """
    sources = get_all_sources()
    results = await search_all(sources, paper_doi, 10 * depth)

    all_papers: list[Paper] = []
    for result in results:
        if result.error is None:
            all_papers.extend(result.papers)

    deduped = _deduplicate(all_papers)
    filtered = [p for p in deduped if p.doi != paper_doi]
    return [_paper_to_dict(p) for p in filtered]


async def expand_topics(seed_topic: str, limit: int = 5) -> list[str]:
    """Generate related query strings derived from the seed topic."""
    return _generate_related_queries(seed_topic, limit)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(mcp) -> None:
    """Register all three search tools onto the given FastMCP instance."""
    mcp.tool()(search_papers)
    mcp.tool()(deep_search)
    mcp.tool()(expand_topics)
