"""SemanticScholarSource — connector for the Semantic Scholar academic API."""

import os
import time

import httpx

from academic_helper.models.paper import Paper
from academic_helper.models.source import SourceResult
from academic_helper.sources.base import AcademicSource

_SOURCE_NAME = "semantic_scholar"
_DEFAULT_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "title,abstract,externalIds,authors,year,citationCount,url"


def _parse_paper(item: dict) -> Paper:
    """Parse a single Semantic Scholar result item into a Paper."""
    authors = tuple(a.get("name", "") for a in (item.get("authors") or []))
    external_ids = item.get("externalIds") or {}
    doi = external_ids.get("DOI")
    return Paper(
        title=item.get("title") or "",
        abstract=item.get("abstract") or "",
        doi=doi,
        authors=authors,
        year=item.get("year"),
        source=_SOURCE_NAME,
        url=item.get("url"),
        citation_count=item.get("citationCount") or 0,
    )


class SemanticScholarSource(AcademicSource):
    """Academic source connector for the Semantic Scholar API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self.base_url = os.environ.get("SEMANTIC_SCHOLAR_BASE_URL", _DEFAULT_BASE_URL)
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self._headers = {"x-api-key": api_key} if api_key else {}

    async def search(self, query: str, limit: int) -> SourceResult:
        """Search Semantic Scholar for papers matching the query."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/paper/search",
                params={"query": query, "limit": limit, "fields": _FIELDS},
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            papers = tuple(_parse_paper(item) for item in data.get("data") or [])
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
        """Fetch a single paper by DOI from Semantic Scholar."""
        try:
            response = await self._client.get(
                f"{self.base_url}/paper/DOI:{doi}",
                params={"fields": _FIELDS},
                headers=self._headers,
            )
            response.raise_for_status()
            return _parse_paper(response.json())
        except httpx.HTTPError:
            return None

    async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers that cite the given DOI from Semantic Scholar."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/paper/DOI:{doi}/citations",
                params={"limit": limit, "fields": _FIELDS},
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            papers = tuple(
                _parse_paper(entry["citingPaper"])
                for entry in data.get("data") or []
                if entry.get("citingPaper")
            )
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, error=str(exc), elapsed_ms=elapsed_ms)

    async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers referenced by the given DOI from Semantic Scholar."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/paper/DOI:{doi}/references",
                params={"limit": limit, "fields": _FIELDS},
                headers=self._headers,
            )
            response.raise_for_status()
            data = response.json()
            papers = tuple(
                _parse_paper(entry["citedPaper"])
                for entry in data.get("data") or []
                if entry.get("citedPaper")
            )
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, error=str(exc), elapsed_ms=elapsed_ms)
