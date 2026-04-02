"""CrossRefSource — connector for the CrossRef academic API."""

import os
import time

import httpx

from academic_helper.models.paper import Paper
from academic_helper.models.source import SourceResult
from academic_helper.sources.base import AcademicSource

_SOURCE_NAME = "crossref"
_DEFAULT_BASE_URL = "https://api.crossref.org"


def _parse_author(author: dict) -> str:
    """Combine given and family name for a CrossRef author entry."""
    given = author.get("given", "")
    family = author.get("family", "")
    return f"{given} {family}".strip()


def _parse_paper(item: dict) -> Paper:
    """Parse a single CrossRef result item into a Paper."""
    titles = item.get("title") or []
    title = titles[0] if titles else ""
    authors = tuple(_parse_author(a) for a in (item.get("author") or []))
    published = item.get("published") or {}
    date_parts = published.get("date-parts") or [[]]
    year_parts = date_parts[0] if date_parts else []
    year = year_parts[0] if year_parts else None
    return Paper(
        title=title,
        abstract=item.get("abstract") or "",
        doi=item.get("DOI"),
        authors=authors,
        year=year,
        source=_SOURCE_NAME,
        url=item.get("URL"),
        citation_count=item.get("is-referenced-by-count") or 0,
    )


class CrossRefSource(AcademicSource):
    """Academic source connector for the CrossRef API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self.base_url = os.environ.get("CROSSREF_BASE_URL", _DEFAULT_BASE_URL)

    async def search(self, query: str, limit: int) -> SourceResult:
        """Search CrossRef for papers matching the query."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/works",
                params={"query": query, "rows": limit},
            )
            response.raise_for_status()
            data = response.json()
            items = (data.get("message") or {}).get("items") or []
            papers = tuple(_parse_paper(item) for item in items)
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(
                source_name=_SOURCE_NAME,
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )

    async def get_paper(self, doi: str) -> Paper | None:
        """Fetch a single paper by DOI from CrossRef."""
        try:
            response = await self._client.get(f"{self.base_url}/works/{doi}")
            response.raise_for_status()
            data = response.json()
            return _parse_paper(data.get("message") or {})
        except httpx.HTTPError:
            return None

    async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers that cite the given DOI from CrossRef."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/works",
                params={"filter": f"references:{doi}", "rows": limit},
            )
            response.raise_for_status()
            data = response.json()
            items = (data.get("message") or {}).get("items") or []
            papers = tuple(_parse_paper(item) for item in items)
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, error=str(exc), elapsed_ms=elapsed_ms)

    async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers referenced by the given DOI from CrossRef."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/works",
                params={"filter": f"references:{doi}", "rows": limit},
            )
            response.raise_for_status()
            data = response.json()
            items = (data.get("message") or {}).get("items") or []
            papers = tuple(_parse_paper(item) for item in items)
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, error=str(exc), elapsed_ms=elapsed_ms)
