"""Extended tests for F004 — AcademicSource ABC with get_paper / get_citations / get_references.

TDD RED phase: all tests in this file MUST FAIL because the three new abstract
methods do not exist yet.  Once the implementation is written (GREEN phase),
every test here should pass.

HTTP calls are fully mocked — no real network traffic occurs.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from academic_helper.models.paper import Paper
from academic_helper.models.source import SourceResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TEST_DOI = "10.1234/test"
_LIMIT = 5


def _make_paper(
    title: str = "Test Paper",
    source: str = "test",
    doi: str | None = _TEST_DOI,
) -> Paper:
    return Paper(
        title=title,
        abstract="Test abstract.",
        doi=doi,
        authors=("Author, A.",),
        year=2024,
        source=source,
        url=f"https://doi.org/{doi}" if doi else None,
        citation_count=42,
    )


def _make_mock_response(json_payload: dict[str, Any]) -> MagicMock:
    """Return a mock httpx.Response that returns the given JSON payload."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock(return_value=None)
    response.json.return_value = json_payload
    return response


def _make_not_found_response() -> MagicMock:
    """Return a mock httpx.Response that raises a 404 HTTPStatusError."""
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found",
        request=MagicMock(),
        response=MagicMock(status_code=404),
    )
    return response


def _make_async_client(response: MagicMock) -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = response
    return client


def _make_error_client(status_code: int = 500) -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = httpx.HTTPStatusError(
        f"{status_code} Error",
        request=MagicMock(),
        response=MagicMock(status_code=status_code),
    )
    return client


def _make_connect_error_client() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = httpx.ConnectError("Connection refused")
    return client


# ---------------------------------------------------------------------------
# Section 1 — AcademicSource ABC extensions
# ---------------------------------------------------------------------------


class TestAcademicSourceABCExtensions:
    """The AcademicSource ABC must declare get_paper, get_citations, and
    get_references as abstract methods."""

    def test_get_paper_is_declared_as_abstract(self):
        from academic_helper.sources.base import AcademicSource

        assert hasattr(AcademicSource, "get_paper"), (
            "AcademicSource must declare get_paper"
        )
        assert getattr(AcademicSource.get_paper, "__isabstractmethod__", False), (
            "get_paper must be an abstract method"
        )

    def test_get_citations_is_declared_as_abstract(self):
        from academic_helper.sources.base import AcademicSource

        assert hasattr(AcademicSource, "get_citations"), (
            "AcademicSource must declare get_citations"
        )
        assert getattr(AcademicSource.get_citations, "__isabstractmethod__", False), (
            "get_citations must be an abstract method"
        )

    def test_get_references_is_declared_as_abstract(self):
        from academic_helper.sources.base import AcademicSource

        assert hasattr(AcademicSource, "get_references"), (
            "AcademicSource must declare get_references"
        )
        assert getattr(AcademicSource.get_references, "__isabstractmethod__", False), (
            "get_references must be an abstract method"
        )

    def test_subclass_missing_get_paper_raises_type_error(self):
        """A subclass that omits get_paper must be uninstantiable."""
        from academic_helper.sources.base import AcademicSource

        class MissingGetPaper(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

        with pytest.raises(TypeError):
            MissingGetPaper()  # type: ignore[abstract]

    def test_subclass_missing_get_citations_raises_type_error(self):
        """A subclass that omits get_citations must be uninstantiable."""
        from academic_helper.sources.base import AcademicSource

        class MissingGetCitations(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_paper(self, doi: str) -> Paper | None:
                return None

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

        with pytest.raises(TypeError):
            MissingGetCitations()  # type: ignore[abstract]

    def test_subclass_missing_get_references_raises_type_error(self):
        """A subclass that omits get_references must be uninstantiable."""
        from academic_helper.sources.base import AcademicSource

        class MissingGetReferences(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_paper(self, doi: str) -> Paper | None:
                return None

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

        with pytest.raises(TypeError):
            MissingGetReferences()  # type: ignore[abstract]

    def test_subclass_implementing_all_methods_is_instantiable(self):
        """A fully-implementing subclass must not raise on instantiation."""
        from academic_helper.sources.base import AcademicSource

        class FullStub(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_paper(self, doi: str) -> Paper | None:
                return None

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

        instance = FullStub()
        assert instance is not None

    @pytest.mark.asyncio
    async def test_get_paper_signature_accepts_doi_returns_paper_or_none(self):
        """get_paper(doi) must accept a str and return Paper | None."""
        from academic_helper.sources.base import AcademicSource

        class Stub(AcademicSource):
            async def search(self, query: str, limit: int) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_paper(self, doi: str) -> Paper | None:
                return _make_paper(source="stub", doi=doi)

            async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

            async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
                return SourceResult(source_name="stub")

        result = await Stub().get_paper(_TEST_DOI)
        assert result is None or isinstance(result, Paper)

    @pytest.mark.asyncio
    async def test_get_citations_signature_has_default_limit(self):
        """get_citations must have limit defaulting to 20."""
        from academic_helper.sources.base import AcademicSource

        sig = inspect.signature(AcademicSource.get_citations)
        limit_param = sig.parameters.get("limit")
        assert limit_param is not None, "get_citations must have a 'limit' parameter"
        assert limit_param.default == 20, "get_citations limit must default to 20"

    @pytest.mark.asyncio
    async def test_get_references_signature_has_default_limit(self):
        """get_references must have limit defaulting to 20."""
        from academic_helper.sources.base import AcademicSource

        sig = inspect.signature(AcademicSource.get_references)
        limit_param = sig.parameters.get("limit")
        assert limit_param is not None, "get_references must have a 'limit' parameter"
        assert limit_param.default == 20, "get_references limit must default to 20"


# ---------------------------------------------------------------------------
# Section 2 — OpenAlexSource extended implementations
# ---------------------------------------------------------------------------


def _openalex_work_item(doi: str = _TEST_DOI, index: int = 1) -> dict[str, Any]:
    return {
        "title": f"OpenAlex Paper {index}",
        "abstract_inverted_index": None,
        "doi": doi,
        "authorships": [{"author": {"display_name": f"Author {index}"}}],
        "publication_year": 2023,
        "cited_by_count": index * 10,
        "primary_location": {"landing_page_url": f"https://doi.org/{doi}"},
    }


def _openalex_single_work_payload(doi: str = _TEST_DOI) -> dict[str, Any]:
    """Payload returned by GET /works/<doi> — a single work object."""
    return _openalex_work_item(doi=doi, index=1)


def _openalex_list_payload(n: int = 3, doi_prefix: str = "10.9999/ref") -> dict[str, Any]:
    """Payload returned by citation/reference list endpoints."""
    return {
        "results": [
            _openalex_work_item(doi=f"{doi_prefix}{i}", index=i)
            for i in range(1, n + 1)
        ]
    }


class TestOpenAlexSourceExtended:
    """Tests for get_paper, get_citations, and get_references on OpenAlexSource."""

    # ---- get_paper --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_paper_returns_paper_on_success(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload(doi=_TEST_DOI)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert isinstance(result, Paper)

    @pytest.mark.asyncio
    async def test_get_paper_maps_title_correctly(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload(doi=_TEST_DOI)
        payload["title"] = "Exact Title Under Test"
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.title == "Exact Title Under Test"

    @pytest.mark.asyncio
    async def test_get_paper_maps_doi_correctly(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload(doi=_TEST_DOI)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.doi == _TEST_DOI

    @pytest.mark.asyncio
    async def test_get_paper_maps_year_correctly(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload(doi=_TEST_DOI)
        payload["publication_year"] = 2021
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.year == 2021

    @pytest.mark.asyncio
    async def test_get_paper_maps_citation_count_correctly(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload(doi=_TEST_DOI)
        payload["cited_by_count"] = 77
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.citation_count == 77

    @pytest.mark.asyncio
    async def test_get_paper_source_tag_is_openalex(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload(doi=_TEST_DOI)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.source == "openalex"

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_404(self):
        from academic_helper.sources.openalex import OpenAlexSource

        client = _make_async_client(_make_not_found_response())
        source = OpenAlexSource(client=client)

        result = await source.get_paper("nonexistent-doi")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_http_error(self):
        from academic_helper.sources.openalex import OpenAlexSource

        source = OpenAlexSource(client=_make_error_client(500))

        result = await source.get_paper(_TEST_DOI)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_connect_error(self):
        from academic_helper.sources.openalex import OpenAlexSource

        source = OpenAlexSource(client=_make_connect_error_client())

        result = await source.get_paper(_TEST_DOI)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_makes_exactly_one_http_call(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_single_work_payload()
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        await source.get_paper(_TEST_DOI)

        client.get.assert_awaited_once()

    # ---- get_citations ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_citations_returns_source_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=_LIMIT)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_get_citations_result_contains_papers(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=_LIMIT)

        assert len(result.papers) == _LIMIT
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_get_citations_source_name_is_openalex(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=1)

        assert result.source_name == "openalex"

    @pytest.mark.asyncio
    async def test_get_citations_http_error_returns_error_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        source = OpenAlexSource(client=_make_error_client(503))

        result = await source.get_citations(_TEST_DOI, limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_citations_connect_error_returns_error_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        source = OpenAlexSource(client=_make_connect_error_client())

        result = await source.get_citations(_TEST_DOI, limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_citations_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=1)

        assert result.elapsed_ms >= 0

    # ---- get_references ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_references_returns_source_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=_LIMIT)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_get_references_result_contains_papers(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=_LIMIT)

        assert len(result.papers) == _LIMIT
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_get_references_source_name_is_openalex(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=1)

        assert result.source_name == "openalex"

    @pytest.mark.asyncio
    async def test_get_references_http_error_returns_error_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        source = OpenAlexSource(client=_make_error_client(503))

        result = await source.get_references(_TEST_DOI, limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_references_connect_error_returns_error_result(self):
        from academic_helper.sources.openalex import OpenAlexSource

        source = OpenAlexSource(client=_make_connect_error_client())

        result = await source.get_references(_TEST_DOI, limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_references_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.openalex import OpenAlexSource

        payload = _openalex_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = OpenAlexSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=1)

        assert result.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# Section 3 — SemanticScholarSource extended implementations
# ---------------------------------------------------------------------------


def _s2_paper_item(doi: str = _TEST_DOI, index: int = 1) -> dict[str, Any]:
    return {
        "title": f"S2 Paper {index}",
        "abstract": f"Abstract {index}",
        "externalIds": {"DOI": doi},
        "authors": [{"name": f"Researcher {index}"}],
        "year": 2022,
        "citationCount": index * 5,
        "url": f"https://www.semanticscholar.org/paper/{index}",
    }


def _s2_single_paper_payload(doi: str = _TEST_DOI) -> dict[str, Any]:
    """Payload returned by the S2 single-paper endpoint."""
    return _s2_paper_item(doi=doi, index=1)


def _s2_list_payload(n: int = 3, key: str = "data") -> dict[str, Any]:
    """Payload for S2 citation/reference list endpoints."""
    return {
        key: [
            {"citedPaper": _s2_paper_item(doi=f"10.5678/s2ref{i}", index=i)}
            if key in ("citations", "references")
            else _s2_paper_item(doi=f"10.5678/s2ref{i}", index=i)
            for i in range(1, n + 1)
        ]
    }


def _s2_citations_payload(n: int = 3) -> dict[str, Any]:
    """S2 citations endpoint wraps each item in a 'citingPaper' key."""
    return {
        "data": [
            {"citingPaper": _s2_paper_item(doi=f"10.5678/s2cit{i}", index=i)}
            for i in range(1, n + 1)
        ]
    }


def _s2_references_payload(n: int = 3) -> dict[str, Any]:
    """S2 references endpoint wraps each item in a 'citedPaper' key."""
    return {
        "data": [
            {"citedPaper": _s2_paper_item(doi=f"10.5678/s2ref{i}", index=i)}
            for i in range(1, n + 1)
        ]
    }


class TestSemanticScholarSourceExtended:
    """Tests for get_paper, get_citations, and get_references on SemanticScholarSource."""

    # ---- get_paper --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_paper_returns_paper_on_success(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_single_paper_payload(doi=_TEST_DOI)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert isinstance(result, Paper)

    @pytest.mark.asyncio
    async def test_get_paper_maps_title_correctly(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_single_paper_payload(doi=_TEST_DOI)
        payload["title"] = "Semantic Scholar Title"
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.title == "Semantic Scholar Title"

    @pytest.mark.asyncio
    async def test_get_paper_maps_abstract_correctly(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_single_paper_payload(doi=_TEST_DOI)
        payload["abstract"] = "Detailed abstract content."
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.abstract == "Detailed abstract content."

    @pytest.mark.asyncio
    async def test_get_paper_source_tag_is_semantic_scholar(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_single_paper_payload()
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.source == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_404(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        client = _make_async_client(_make_not_found_response())
        source = SemanticScholarSource(client=client)

        result = await source.get_paper("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_http_error(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource(client=_make_error_client(500))

        result = await source.get_paper(_TEST_DOI)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_connect_error(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource(client=_make_connect_error_client())

        result = await source.get_paper(_TEST_DOI)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_makes_exactly_one_http_call(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_single_paper_payload()
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        await source.get_paper(_TEST_DOI)

        client.get.assert_awaited_once()

    # ---- get_citations ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_citations_returns_source_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_citations_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=_LIMIT)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_get_citations_result_contains_papers(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_citations_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=_LIMIT)

        assert len(result.papers) == _LIMIT
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_get_citations_source_name_is_semantic_scholar(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_citations_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=1)

        assert result.source_name == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_get_citations_http_error_returns_error_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource(client=_make_error_client(429))

        result = await source.get_citations(_TEST_DOI, limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_citations_connect_error_returns_error_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource(client=_make_connect_error_client())

        result = await source.get_citations(_TEST_DOI, limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_citations_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_citations_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=1)

        assert result.elapsed_ms >= 0

    # ---- get_references ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_references_returns_source_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_references_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=_LIMIT)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_get_references_result_contains_papers(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_references_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=_LIMIT)

        assert len(result.papers) == _LIMIT
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_get_references_source_name_is_semantic_scholar(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_references_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=1)

        assert result.source_name == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_get_references_http_error_returns_error_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource(client=_make_error_client(500))

        result = await source.get_references(_TEST_DOI, limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_references_connect_error_returns_error_result(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        source = SemanticScholarSource(client=_make_connect_error_client())

        result = await source.get_references(_TEST_DOI, limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_references_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.semantic_scholar import SemanticScholarSource

        payload = _s2_references_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = SemanticScholarSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=1)

        assert result.elapsed_ms >= 0


# ---------------------------------------------------------------------------
# Section 4 — CrossRefSource extended implementations
# ---------------------------------------------------------------------------


def _crossref_work_item(doi: str = _TEST_DOI, index: int = 1) -> dict[str, Any]:
    return {
        "title": [f"CrossRef Paper {index}"],
        "abstract": f"<jats:p>Abstract {index}</jats:p>",
        "DOI": doi,
        "author": [{"given": f"First{index}", "family": f"Last{index}"}],
        "published": {"date-parts": [[2021]]},
        "is-referenced-by-count": index * 3,
        "URL": f"https://doi.org/{doi}",
    }


def _crossref_single_work_payload(doi: str = _TEST_DOI) -> dict[str, Any]:
    """Payload for GET /works/<doi> — wrapped in the CrossRef message envelope."""
    return {"message": _crossref_work_item(doi=doi, index=1)}


def _crossref_list_payload(n: int = 3) -> dict[str, Any]:
    """Payload for CrossRef list endpoints (no separate citations/references endpoints)."""
    return {
        "message": {
            "items": [
                _crossref_work_item(doi=f"10.9012/cr{i}", index=i)
                for i in range(1, n + 1)
            ]
        }
    }


class TestCrossRefSourceExtended:
    """Tests for get_paper, get_citations, and get_references on CrossRefSource."""

    # ---- get_paper --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_paper_returns_paper_on_success(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_single_work_payload(doi=_TEST_DOI)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert isinstance(result, Paper)

    @pytest.mark.asyncio
    async def test_get_paper_maps_title_correctly(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_single_work_payload(doi=_TEST_DOI)
        payload["message"]["title"] = ["CrossRef Exact Title"]
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.title == "CrossRef Exact Title"

    @pytest.mark.asyncio
    async def test_get_paper_maps_doi_correctly(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_single_work_payload(doi=_TEST_DOI)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.doi == _TEST_DOI

    @pytest.mark.asyncio
    async def test_get_paper_maps_year_correctly(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_single_work_payload(doi=_TEST_DOI)
        payload["message"]["published"] = {"date-parts": [[2019]]}
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.year == 2019

    @pytest.mark.asyncio
    async def test_get_paper_source_tag_is_crossref(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_single_work_payload()
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_paper(_TEST_DOI)

        assert result is not None
        assert result.source == "crossref"

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_404(self):
        from academic_helper.sources.crossref import CrossRefSource

        client = _make_async_client(_make_not_found_response())
        source = CrossRefSource(client=client)

        result = await source.get_paper("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_http_error(self):
        from academic_helper.sources.crossref import CrossRefSource

        source = CrossRefSource(client=_make_error_client(500))

        result = await source.get_paper(_TEST_DOI)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_returns_none_on_connect_error(self):
        from academic_helper.sources.crossref import CrossRefSource

        source = CrossRefSource(client=_make_connect_error_client())

        result = await source.get_paper(_TEST_DOI)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_paper_makes_exactly_one_http_call(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_single_work_payload()
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        await source.get_paper(_TEST_DOI)

        client.get.assert_awaited_once()

    # ---- get_citations ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_citations_returns_source_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=_LIMIT)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_get_citations_result_contains_papers(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=_LIMIT)

        assert len(result.papers) == _LIMIT
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_get_citations_source_name_is_crossref(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=1)

        assert result.source_name == "crossref"

    @pytest.mark.asyncio
    async def test_get_citations_http_error_returns_error_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        source = CrossRefSource(client=_make_error_client(500))

        result = await source.get_citations(_TEST_DOI, limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_citations_connect_error_returns_error_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        source = CrossRefSource(client=_make_connect_error_client())

        result = await source.get_citations(_TEST_DOI, limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_citations_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_citations(_TEST_DOI, limit=1)

        assert result.elapsed_ms >= 0

    # ---- get_references ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_references_returns_source_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=_LIMIT)

        assert isinstance(result, SourceResult)

    @pytest.mark.asyncio
    async def test_get_references_result_contains_papers(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=_LIMIT)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=_LIMIT)

        assert len(result.papers) == _LIMIT
        assert all(isinstance(p, Paper) for p in result.papers)

    @pytest.mark.asyncio
    async def test_get_references_source_name_is_crossref(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=1)

        assert result.source_name == "crossref"

    @pytest.mark.asyncio
    async def test_get_references_http_error_returns_error_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        source = CrossRefSource(client=_make_error_client(503))

        result = await source.get_references(_TEST_DOI, limit=5)

        assert isinstance(result, SourceResult)
        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_references_connect_error_returns_error_result(self):
        from academic_helper.sources.crossref import CrossRefSource

        source = CrossRefSource(client=_make_connect_error_client())

        result = await source.get_references(_TEST_DOI, limit=5)

        assert result.error is not None
        assert result.papers == ()

    @pytest.mark.asyncio
    async def test_get_references_elapsed_ms_is_non_negative(self):
        from academic_helper.sources.crossref import CrossRefSource

        payload = _crossref_list_payload(n=1)
        client = _make_async_client(_make_mock_response(payload))
        source = CrossRefSource(client=client)

        result = await source.get_references(_TEST_DOI, limit=1)

        assert result.elapsed_ms >= 0
