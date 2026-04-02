"""Tests for academic API source connectors — TDD RED phase.

These tests define the expected interface and behaviour of:
  - AcademicSource  (ABC)
  - OpenAlexSource
  - SemanticScholarSource
  - CrossRefSource
  - registry.get_all_sources()
  - search_all() concurrent helper

All HTTP calls are mocked; no real network traffic occurs.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from academic_helper.models.paper import Paper
from academic_helper.models.source import SourceResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper(
    title: str = "Test Paper",
    source: str = "test",
    doi: str | None = "10.0000/test",
) -> Paper:
    return Paper(
        title=title,
        abstract="Abstract text.",
        doi=doi,
        authors=("Author, A.",),
        year=2024,
        source=source,
        url=f"https://doi.org/{doi}" if doi else None,
        citation_count=0,
    )


# ---------------------------------------------------------------------------
# AcademicSource ABC
# ---------------------------------------------------------------------------


class TestAcademicSourceABC:
    """AcademicSource must be an abstract base class with a search() method."""

    def test_academic_source_is_importable(self):
        from academic_helper.sources.base import AcademicSource  # noqa: F401

    def test_academic_source_is_abstract(self):
        from academic_helper.sources.base import AcademicSource
        import inspect

        assert inspect.isabstract(AcademicSource)

    def test_academic_source_cannot_be_instantiated_directly(self):
        from academic_helper.sources.base import AcademicSource

        with pytest.raises(TypeError):
            AcademicSource()  # type: ignore[abstract]

    def test_academic_source_has_abstract_search_method(self):
        from academic_helper.sources.base import AcademicSource

        assert hasattr(AcademicSource, "search")
        assert getattr(AcademicSource.search, "__isabstractmethod__", False)

    @pytest.mark.asyncio
    async def test_concrete_subclass_must_implement_search(self):
        """A subclass that omits search() must also be uninstantiable."""
        from academic_helper.sources.base import AcademicSource

        class Incomplete(AcademicSource):
            # no search() implementation — deliberately incomplete
            async def get_paper(self, doi: str) -> Paper | None:
                return None

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub", papers=())

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub", papers=())

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    @pytest.mark.asyncio
    async def test_search_signature_returns_source_result(self):
        """A minimal concrete subclass that returns a SourceResult must work."""
        from academic_helper.sources.base import AcademicSource

        class Stub(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                return SourceResult(source_name="stub", papers=())

            async def get_paper(self, doi: str) -> Paper | None:
                return None

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub", papers=())

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub", papers=())

        stub = Stub()
        result = await stub.search("nursing AI", limit=5)
        assert isinstance(result, SourceResult)


# ---------------------------------------------------------------------------
# OpenAlexSource
# ---------------------------------------------------------------------------


class TestOpenAlexSource:
    """Tests for the OpenAlex API connector."""

    def _make_client(self, json_response: dict[str, Any] | None = None) -> httpx.AsyncClient:
        """Return a mock AsyncClient whose get() returns the given JSON payload."""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock(return_value=None)
        if json_response is not None:
            mock_response.json.return_value = json_response
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response
        return client

    def _openalex_payload(self, n: int = 2) -> dict[str, Any]:
        return {
            "results": [
                {
                    "title": f"OpenAlex Paper {i}",
                    "abstract_inverted_index": None,
                    "doi": f"10.1234/oa{i}",
                    "authorships": [{"author": {"display_name": f"Author {i}"}}],
                    "publication_year": 2023,
                    "cited_by_count": i * 10,
                    "primary_location": {"landing_page_url": f"https://doi.org/10.1234/oa{i}"},
                }
                for i in range(1, n + 1)
            ]
        }

    def test_openalex_source_is_importable(self):
        from academic_helper.sources.openalex import OpenAlexSource  # noqa: F401

    def test_openalex_source_is_subclass_of_academic_source(self):
        from academic_helper.sources.base import AcademicSource
        from academic_helper.sources.openalex import OpenAlexSource

        assert issubclass(OpenAlexSource, AcademicSource)

    def test_openalex_source_accepts_http_client(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = AsyncMock(spec=httpx.AsyncClient)
        source = OpenAlexSource(client=client)
        assert source is not None

    @pytest.mark.asyncio
    async def test_openalex_search_returns_source_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = self._make_client(self._openalex_payload(2))
        source = OpenAlexSource(client=client)
        result = await source.search("nursing AI elderly", limit=2)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_openalex_search_result_contains_papers(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = self._make_client(self._openalex_payload(2))
        source = OpenAlexSource(client=client)
        result = await source.search("nursing AI elderly", limit=2)

        assert len(result.papers) == 2
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_openalex_papers_have_correct_source_tag(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = self._make_client(self._openalex_payload(1))
        source = OpenAlexSource(client=client)
        result = await source.search("nursing", limit=1)

        assert result.papers[0].source == "openalex"

    @pytest.mark.asyncio
    async def test_openalex_source_name_is_set(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = self._make_client(self._openalex_payload(1))
        source = OpenAlexSource(client=client)
        result = await source.search("query", limit=5)

        assert result.source_name == "openalex"

    @pytest.mark.asyncio
    async def test_openalex_no_real_http_call_is_made(self):
        """The mock client's get() must be invoked, not any real network call."""
        from academic_helper.sources.openalex import OpenAlexSource

        client = self._make_client(self._openalex_payload(1))
        source = OpenAlexSource(client=client)
        await source.search("query", limit=1)

        client.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_openalex_http_error_returns_error_result(self):
        """HTTPStatusError must be caught; error field must be set."""
        from academic_helper.sources.openalex import OpenAlexSource

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        source = OpenAlexSource(client=client)
        result = await source.search("query", limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_openalex_connection_error_returns_error_result(self):
        """Network-level errors must also be caught."""
        from academic_helper.sources.openalex import OpenAlexSource

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ConnectError("Connection refused")
        source = OpenAlexSource(client=client)
        result = await source.search("query", limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_openalex_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = self._make_client(self._openalex_payload(1))
        source = OpenAlexSource(client=client)
        result = await source.search("query", limit=5)

        assert result.elapsed_ms >= 0

    def test_openalex_uses_env_var_for_base_url(self, monkeypatch):
        """Base URL should default to an OpenAlex endpoint and be overridable."""
        from academic_helper.sources.openalex import OpenAlexSource

        monkeypatch.setenv("OPENALEX_BASE_URL", "https://custom.openalex.example/")
        client = AsyncMock(spec=httpx.AsyncClient)
        source = OpenAlexSource(client=client)
        assert "custom.openalex.example" in source.base_url


# ---------------------------------------------------------------------------
# SemanticScholarSource
# ---------------------------------------------------------------------------


class TestSemanticScholarSource:
    """Tests for the Semantic Scholar API connector."""

    def _make_client(self, json_response: dict[str, Any] | None = None) -> httpx.AsyncClient:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock(return_value=None)
        if json_response is not None:
            mock_response.json.return_value = json_response
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response
        return client

    def _s2_payload(self, n: int = 2) -> dict[str, Any]:
        return {
            "data": [
                {
                    "title": f"S2 Paper {i}",
                    "abstract": f"Abstract {i}",
                    "externalIds": {"DOI": f"10.5678/s2{i}"},
                    "authors": [{"name": f"Researcher {i}"}],
                    "year": 2022,
                    "citationCount": i * 5,
                    "url": f"https://www.semanticscholar.org/paper/{i}",
                }
                for i in range(1, n + 1)
            ]
        }

    def test_semantic_scholar_source_is_importable(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource  # noqa: F401

    def test_semantic_scholar_source_is_subclass_of_academic_source(self):
        from academic_helper.sources.base import AcademicSource
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        assert issubclass(SemanticScholarSource, AcademicSource)

    def test_semantic_scholar_accepts_http_client(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = AsyncMock(spec=httpx.AsyncClient)
        source = SemanticScholarSource(client=client)
        assert source is not None

    @pytest.mark.asyncio
    async def test_semantic_scholar_search_returns_source_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = self._make_client(self._s2_payload(2))
        source = SemanticScholarSource(client=client)
        result = await source.search("machine learning", limit=2)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_semantic_scholar_search_result_contains_papers(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = self._make_client(self._s2_payload(2))
        source = SemanticScholarSource(client=client)
        result = await source.search("machine learning", limit=2)

        assert len(result.papers) == 2
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_semantic_scholar_papers_have_correct_source_tag(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = self._make_client(self._s2_payload(1))
        source = SemanticScholarSource(client=client)
        result = await source.search("nursing", limit=1)

        assert result.papers[0].source == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_semantic_scholar_source_name_is_set(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = self._make_client(self._s2_payload(1))
        source = SemanticScholarSource(client=client)
        result = await source.search("query", limit=5)

        assert result.source_name == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_semantic_scholar_http_error_returns_error_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=MagicMock(),
            response=MagicMock(status_code=503),
        )
        source = SemanticScholarSource(client=client)
        result = await source.search("query", limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_semantic_scholar_connection_error_returns_error_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ConnectError("Connection refused")
        source = SemanticScholarSource(client=client)
        result = await source.search("query", limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_semantic_scholar_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = self._make_client(self._s2_payload(1))
        source = SemanticScholarSource(client=client)
        result = await source.search("query", limit=5)

        assert result.elapsed_ms >= 0

    def test_semantic_scholar_uses_env_var_for_base_url(self, monkeypatch):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        monkeypatch.setenv("SEMANTIC_SCHOLAR_BASE_URL", "https://custom.s2.example/")
        client = AsyncMock(spec=httpx.AsyncClient)
        source = SemanticScholarSource(client=client)
        assert "custom.s2.example" in source.base_url


# ---------------------------------------------------------------------------
# CrossRefSource
# ---------------------------------------------------------------------------


class TestCrossRefSource:
    """Tests for the CrossRef API connector."""

    def _make_client(self, json_response: dict[str, Any] | None = None) -> httpx.AsyncClient:
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status = MagicMock(return_value=None)
        if json_response is not None:
            mock_response.json.return_value = json_response
        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.return_value = mock_response
        return client

    def _crossref_payload(self, n: int = 2) -> dict[str, Any]:
        return {
            "message": {
                "items": [
                    {
                        "title": [f"CrossRef Paper {i}"],
                        "abstract": f"<jats:p>Abstract {i}</jats:p>",
                        "DOI": f"10.9012/cr{i}",
                        "author": [{"given": f"First{i}", "family": f"Last{i}"}],
                        "published": {"date-parts": [[2021]]},
                        "is-referenced-by-count": i * 3,
                        "URL": f"https://doi.org/10.9012/cr{i}",
                    }
                    for i in range(1, n + 1)
                ]
            }
        }

    def test_crossref_source_is_importable(self):
        from academic_helper.sources.crossref import CrossRefSource  # noqa: F401

    def test_crossref_source_is_subclass_of_academic_source(self):
        from academic_helper.sources.base import AcademicSource
        from academic_helper.sources.crossref import CrossRefSource

        assert issubclass(CrossRefSource, AcademicSource)

    def test_crossref_accepts_http_client(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = AsyncMock(spec=httpx.AsyncClient)
        source = CrossRefSource(client=client)
        assert source is not None

    @pytest.mark.asyncio
    async def test_crossref_search_returns_source_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = self._make_client(self._crossref_payload(2))
        source = CrossRefSource(client=client)
        result = await source.search("systematic review", limit=2)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_crossref_search_result_contains_papers(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = self._make_client(self._crossref_payload(2))
        source = CrossRefSource(client=client)
        result = await source.search("systematic review", limit=2)

        assert len(result.papers) == 2
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_crossref_papers_have_correct_source_tag(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = self._make_client(self._crossref_payload(1))
        source = CrossRefSource(client=client)
        result = await source.search("nursing", limit=1)

        assert result.papers[0].source == "crossref"

    @pytest.mark.asyncio
    async def test_crossref_source_name_is_set(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = self._make_client(self._crossref_payload(1))
        source = CrossRefSource(client=client)
        result = await source.search("query", limit=5)

        assert result.source_name == "crossref"

    @pytest.mark.asyncio
    async def test_crossref_http_error_returns_error_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        source = CrossRefSource(client=client)
        result = await source.search("query", limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_crossref_connection_error_returns_error_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get.side_effect = httpx.ConnectError("Connection refused")
        source = CrossRefSource(client=client)
        result = await source.search("query", limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_crossref_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = self._make_client(self._crossref_payload(1))
        source = CrossRefSource(client=client)
        result = await source.search("query", limit=5)

        assert result.elapsed_ms >= 0

    def test_crossref_uses_env_var_for_base_url(self, monkeypatch):
        from academic_helper.sources.crossref import CrossRefSource

        monkeypatch.setenv("CROSSREF_BASE_URL", "https://custom.crossref.example/")
        client = AsyncMock(spec=httpx.AsyncClient)
        source = CrossRefSource(client=client)
        assert "custom.crossref.example" in source.base_url


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the source registry that returns all connectors."""

    def test_registry_is_importable(self):
        from academic_helper.sources.registry import get_all_sources  # noqa: F401

    def test_get_all_sources_returns_tuple(self):
        from academic_helper.sources.registry import get_all_sources

        sources = get_all_sources()
        assert isinstance(sources, tuple)

    def test_get_all_sources_returns_three_connectors(self):
        from academic_helper.sources.registry import get_all_sources

        sources = get_all_sources()
        assert len(sources) == 3

    def test_get_all_sources_contains_all_three_types(self):
        from academic_helper.sources.crossref import CrossRefSource
        from academic_helper.sources.openalex import OpenAlexSource
        from academic_helper.sources.registry import get_all_sources
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        sources = get_all_sources()
        source_types = {type(s) for s in sources}
        assert OpenAlexSource in source_types
        assert SemanticScholarSource in source_types
        assert CrossRefSource in source_types

    def test_get_all_sources_each_is_academic_source(self):
        from academic_helper.sources.base import AcademicSource
        from academic_helper.sources.registry import get_all_sources

        for source in get_all_sources():
            assert isinstance(source, AcademicSource)

    def test_get_all_sources_is_idempotent(self):
        """Calling get_all_sources() twice must return distinct tuple objects
        (no shared mutable state) but with equivalent connector types."""
        from academic_helper.sources.registry import get_all_sources

        first = get_all_sources()
        second = get_all_sources()
        assert type(first[0]) is type(second[0])
        # They must not be the exact same tuple (fresh instances each call)
        assert first is not second


# ---------------------------------------------------------------------------
# search_all — concurrent helper
# ---------------------------------------------------------------------------


class TestSearchAll:
    """Tests for the search_all() concurrent search helper."""

    def test_search_all_is_importable(self):
        from academic_helper.sources.search import search_all  # noqa: F401

    @pytest.mark.asyncio
    async def test_search_all_returns_tuple(self):
        from academic_helper.sources.search import search_all

        sources = self._make_three_stub_sources(error=False)
        results = await search_all(sources, "nursing AI", limit=5)

        assert isinstance(results, tuple)

    @pytest.mark.asyncio
    async def test_search_all_returns_one_result_per_source(self):
        from academic_helper.sources.search import search_all

        sources = self._make_three_stub_sources(error=False)
        results = await search_all(sources, "nursing AI", limit=5)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_all_results_are_source_results(self):
        from academic_helper.sources.search import search_all

        sources = self._make_three_stub_sources(error=False)
        results = await search_all(sources, "nursing AI", limit=5)

        assert all(isinstance(r, SourceResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_all_uses_asyncio_gather(self):
        """search_all must call asyncio.gather (concurrent execution)."""
        from academic_helper.sources import search as search_module

        sources = self._make_three_stub_sources(error=False)
        with patch.object(asyncio, "gather", wraps=asyncio.gather) as mock_gather:
            await search_module.search_all(sources, "query", limit=3)

        mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_all_one_error_does_not_stop_others(self):
        """A fault in one source must not prevent the remaining two from returning."""
        from academic_helper.sources.search import search_all

        sources = self._make_three_stub_sources(error=True, error_index=1)
        results = await search_all(sources, "query", limit=5)

        assert len(results) == 3

        # The non-failing sources return papers
        assert results[0].error is None
        assert len(results[0].papers) > 0
        assert results[2].error is None
        assert len(results[2].papers) > 0

        # The failing source has an error set and no papers
        assert results[1].error is not None
        assert results[1].papers == ()

    @pytest.mark.asyncio
    async def test_search_all_empty_sources_returns_empty_tuple(self):
        from academic_helper.sources.search import search_all

        results = await search_all((), "query", limit=5)
        assert results == ()

    @pytest.mark.asyncio
    async def test_search_all_passes_query_and_limit_to_each_source(self):
        from academic_helper.sources.search import search_all

        sources = self._make_three_stub_sources(error=False)
        await search_all(sources, "specific query", limit=7)

        for source in sources:
            source.search.assert_awaited_once_with("specific query", 7)

    # ------------------------------------------------------------------
    # Helpers for TestSearchAll
    # ------------------------------------------------------------------

    def _make_stub_source(
        self,
        name: str,
        error: bool = False,
    ):
        """Return an AcademicSource stub that either succeeds or fails."""
        from academic_helper.sources.base import AcademicSource

        class StubSource(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                if error:
                    return SourceResult(source_name=name, error="Simulated error")
                return SourceResult(
                    source_name=name,
                    papers=(_make_paper(title=f"Paper from {name}", source=name),),
                )

            async def get_paper(self, doi: str) -> Paper | None:
                return None

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name=name, papers=())

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name=name, papers=())

        stub = StubSource()
        # Wrap search in AsyncMock so we can assert await calls
        original_search = stub.search

        async def tracked_search(query: str, limit: int) -> SourceResult:
            return await original_search(query, limit)

        stub.search = AsyncMock(side_effect=tracked_search)
        return stub

    def _make_three_stub_sources(
        self,
        error: bool,
        error_index: int | None = None,
    ) -> tuple:
        names = ("openalex", "semantic_scholar", "crossref")
        return tuple(
            self._make_stub_source(
                name=name,
                error=(error and error_index is None) or (error_index is not None and i == error_index),
            )
            for i, name in enumerate(names)
        )


# ---------------------------------------------------------------------------
# Fault Isolation — cross-source scenario
# ---------------------------------------------------------------------------


class TestFaultIsolation:
    """Integration-style tests verifying fault isolation across all connectors."""

    @pytest.mark.asyncio
    async def test_openalex_error_does_not_affect_semantic_scholar(self):
        """An HTTP error in OpenAlex must not infect SemanticScholar's result."""
        from academic_helper.sources.openalex import OpenAlexSource
        from academic_helper.sources.search import search_all
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        failing_client = AsyncMock(spec=httpx.AsyncClient)
        failing_client.get.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=MagicMock(status_code=429)
        )
        oa_source = OpenAlexSource(client=failing_client)

        s2_response = MagicMock(spec=httpx.Response)
        s2_response.raise_for_status = MagicMock(return_value=None)
        s2_response.json.return_value = {
            "data": [
                {
                    "title": "Good S2 Paper",
                    "abstract": "Abstract",
                    "externalIds": {"DOI": "10.5678/good"},
                    "authors": [{"name": "Author A"}],
                    "year": 2024,
                    "citationCount": 5,
                    "url": "https://www.semanticscholar.org/paper/1",
                }
            ]
        }
        good_client = AsyncMock(spec=httpx.AsyncClient)
        good_client.get.return_value = s2_response
        s2_source = SemanticScholarSource(client=good_client)

        results = await search_all((oa_source, s2_source), "query", limit=5)

        oa_result = next(r for r in results if r.source_name == "openalex")
        s2_result = next(r for r in results if r.source_name == "semantic_scholar")

        assert oa_result.error is not None
        assert oa_result.papers == ()

        assert s2_result.error is None
        assert len(s2_result.papers) == 1

    @pytest.mark.asyncio
    async def test_all_three_sources_fail_independently(self):
        """When all three connectors fail, each SourceResult has its own error."""
        from academic_helper.sources.crossref import CrossRefSource
        from academic_helper.sources.openalex import OpenAlexSource
        from academic_helper.sources.search import search_all
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        def make_failing_client(status_code: int) -> httpx.AsyncClient:
            client = AsyncMock(spec=httpx.AsyncClient)
            client.get.side_effect = httpx.HTTPStatusError(
                f"{status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=status_code),
            )
            return client

        sources = (
            OpenAlexSource(client=make_failing_client(429)),
            SemanticScholarSource(client=make_failing_client(503)),
            CrossRefSource(client=make_failing_client(500)),
        )
        results = await search_all(sources, "query", limit=5)

        assert len(results) == 3
        for result in results:
            assert result.error is not None
            assert result.papers == ()
