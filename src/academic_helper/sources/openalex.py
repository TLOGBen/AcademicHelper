"""OpenAlexSource — connector for the OpenAlex academic API."""

import os
import time

import httpx

from academic_helper.models.paper import Paper
from academic_helper.models.source import SourceResult
from academic_helper.sources.base import AcademicSource

_SOURCE_NAME = "openalex"
_DEFAULT_BASE_URL = "https://api.openalex.org"


def _parse_paper(item: dict) -> Paper:
    """Parse a single OpenAlex result item into a Paper."""
    authorships = item.get("authorships") or []
    authors = tuple(
        a["author"]["display_name"]
        for a in authorships
        if a.get("author", {}).get("display_name")
    )
    primary_location = item.get("primary_location") or {}
    url = primary_location.get("landing_page_url")
    return Paper(
        title=item.get("title") or "",
        abstract="",
        doi=item.get("doi"),
        authors=authors,
        year=item.get("publication_year"),
        source=_SOURCE_NAME,
        url=url,
        citation_count=item.get("cited_by_count") or 0,
    )


class OpenAlexSource(AcademicSource):
    """Academic source connector for the OpenAlex API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self.base_url = os.environ.get("OPENALEX_BASE_URL", _DEFAULT_BASE_URL)

    async def search(self, query: str, limit: int) -> SourceResult:
        """Search OpenAlex for papers matching the query."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/works",
                params={"search": query, "per_page": limit, "sort": "relevance_score:desc"},
            )
            response.raise_for_status()
            data = response.json()
            papers = tuple(_parse_paper(item) for item in data.get("results") or [])
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
        """Fetch a single paper by DOI from OpenAlex."""
        try:
            response = await self._client.get(f"{self.base_url}/works/doi:{doi}")
            response.raise_for_status()
            return _parse_paper(response.json())
        except httpx.HTTPError:
            return None

    async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers that cite the given DOI from OpenAlex."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/works",
                params={"filter": f"cites:doi:{doi}", "per_page": limit},
            )
            response.raise_for_status()
            data = response.json()
            papers = tuple(_parse_paper(item) for item in data.get("results") or [])
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, error=str(exc), elapsed_ms=elapsed_ms)

    async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers referenced by the given DOI from OpenAlex."""
        start = time.monotonic_ns()
        try:
            response = await self._client.get(
                f"{self.base_url}/works",
                params={"filter": f"cited_by:doi:{doi}", "per_page": limit},
            )
            response.raise_for_status()
            data = response.json()
            papers = tuple(_parse_paper(item) for item in data.get("results") or [])
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, papers=papers, elapsed_ms=elapsed_ms)
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            return SourceResult(source_name=_SOURCE_NAME, error=str(exc), elapsed_ms=elapsed_ms)
