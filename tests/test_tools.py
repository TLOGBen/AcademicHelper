"""Tests for MCP search tools — TDD RED phase (L004).

Covers:
  - academic_helper.tools.search  (register, search_papers, deep_search, expand_topics)
  - academic_helper.tools.pdf     (extract_paper_from_pdf)

No real HTTP calls or file I/O are made; all external dependencies are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
    citation_count: int = 0,
    year: int = 2024,
) -> Paper:
    return Paper(
        title=title,
        abstract="Abstract text.",
        doi=doi,
        authors=("Author, A.",),
        year=year,
        source=source,
        url=f"https://doi.org/{doi}" if doi else None,
        citation_count=citation_count,
    )


def _make_source_result(
    source_name: str,
    papers: tuple[Paper, ...] = (),
    error: str | None = None,
) -> SourceResult:
    return SourceResult(source_name=source_name, papers=papers, error=error)


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestSearchToolsImportable:
    """The tools.search module and register() must be importable."""

    def test_search_module_is_importable(self):
        from academic_helper.tools import search  # noqa: F401

    def test_register_function_exists(self):
        from academic_helper.tools.search import register

        assert callable(register)

    def test_search_papers_function_exists(self):
        from academic_helper.tools.search import search_papers

        assert callable(search_papers)

    def test_deep_search_function_exists(self):
        from academic_helper.tools.search import deep_search

        assert callable(deep_search)

    def test_expand_topics_function_exists(self):
        from academic_helper.tools.search import expand_topics

        assert callable(expand_topics)


class TestPdfToolsImportable:
    """The tools.pdf module must be importable."""

    def test_pdf_module_is_importable(self):
        from academic_helper.tools import pdf  # noqa: F401

    def test_extract_paper_from_pdf_function_exists(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        assert callable(extract_paper_from_pdf)


# ---------------------------------------------------------------------------
# register() — MCP tool registration
# ---------------------------------------------------------------------------


class TestRegisterSearchTools:
    """register(mcp) must register all three tool functions on a FastMCP instance."""

    def _make_mock_mcp(self) -> MagicMock:
        """Return a mock FastMCP that records tool() decorator calls."""
        mock_mcp = MagicMock()
        # tool() is a decorator factory — it must return a callable that accepts a fn
        mock_mcp.tool.return_value = lambda fn: fn
        return mock_mcp

    def test_register_calls_tool_decorator_on_mcp(self):
        from academic_helper.tools.search import register

        mock_mcp = self._make_mock_mcp()
        register(mock_mcp)

        assert mock_mcp.tool.called, "register() must call mcp.tool() at least once"

    def test_register_registers_three_tools(self):
        from academic_helper.tools.search import register

        mock_mcp = self._make_mock_mcp()
        register(mock_mcp)

        assert mock_mcp.tool.call_count == 3, (
            f"Expected 3 tool registrations, got {mock_mcp.tool.call_count}"
        )

    def test_register_with_real_fastmcp_does_not_raise(self):
        from mcp.server.fastmcp import FastMCP

        from academic_helper.tools.search import register

        test_mcp = FastMCP("test-register")
        # Must not raise
        register(test_mcp)


# ---------------------------------------------------------------------------
# search_papers — deduplication and basic behaviour
# ---------------------------------------------------------------------------


class TestSearchPapers:
    """search_papers(query, limit) calls search_all() and deduplicates by DOI."""

    @pytest.mark.asyncio
    async def test_search_papers_returns_list(self):
        from academic_helper.tools.search import search_papers

        paper = _make_paper(doi="10.1234/a", citation_count=5)
        result_a = _make_source_result("openalex", papers=(paper,))
        result_b = _make_source_result("crossref", papers=())

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(result_a, result_b)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(), MagicMock()),
        ):
            results = await search_papers("nursing AI", limit=10)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_papers_returns_list_of_dicts(self):
        from academic_helper.tools.search import search_papers

        paper = _make_paper(doi="10.1234/a", citation_count=3)
        result = _make_source_result("openalex", papers=(paper,))

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(result,)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(),),
        ):
            results = await search_papers("nursing AI", limit=10)

        assert all(isinstance(r, dict) for r in results)

    @pytest.mark.asyncio
    async def test_search_papers_deduplicates_by_doi(self):
        """Same DOI appearing in two SourceResults should appear only once."""
        from academic_helper.tools.search import search_papers

        doi = "10.0000/shared"
        paper_a = _make_paper(title="From OA", doi=doi, source="openalex", citation_count=10)
        paper_b = _make_paper(title="From S2", doi=doi, source="semantic_scholar", citation_count=5)

        result_a = _make_source_result("openalex", papers=(paper_a,))
        result_b = _make_source_result("semantic_scholar", papers=(paper_b,))

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(result_a, result_b)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(), MagicMock()),
        ):
            results = await search_papers("query", limit=10)

        dois_in_result = [r["doi"] for r in results if r.get("doi") == doi]
        assert len(dois_in_result) == 1, (
            "Duplicate DOI must appear exactly once after deduplication"
        )

    @pytest.mark.asyncio
    async def test_search_papers_dedup_keeps_higher_citation_count(self):
        """When deduplicating, keep the paper with the higher citation_count."""
        from academic_helper.tools.search import search_papers

        doi = "10.0000/shared"
        low_cite = _make_paper(title="Low", doi=doi, source="crossref", citation_count=2)
        high_cite = _make_paper(title="High", doi=doi, source="openalex", citation_count=99)

        result_a = _make_source_result("crossref", papers=(low_cite,))
        result_b = _make_source_result("openalex", papers=(high_cite,))

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(result_a, result_b)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(), MagicMock()),
        ):
            results = await search_papers("query", limit=10)

        matched = [r for r in results if r.get("doi") == doi]
        assert len(matched) == 1
        assert matched[0]["citation_count"] == 99, (
            "Deduplication must keep the paper with the higher citation_count"
        )

    @pytest.mark.asyncio
    async def test_search_papers_includes_papers_without_doi(self):
        """Papers with doi=None must still be included (no deduplication applied)."""
        from academic_helper.tools.search import search_papers

        paper_no_doi_a = _make_paper(title="No DOI A", doi=None, source="openalex")
        paper_no_doi_b = _make_paper(title="No DOI B", doi=None, source="crossref")

        result = _make_source_result(
            "openalex", papers=(paper_no_doi_a, paper_no_doi_b)
        )

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(result,)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(),),
        ):
            results = await search_papers("query", limit=10)

        titles = [r["title"] for r in results]
        assert "No DOI A" in titles
        assert "No DOI B" in titles

    @pytest.mark.asyncio
    async def test_search_papers_passes_query_to_search_all(self):
        from academic_helper.tools.search import search_papers

        mock_search_all = AsyncMock(return_value=())
        with patch(
            "academic_helper.tools.search.search_all",
            new=mock_search_all,
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(),
        ):
            await search_papers("specific query text", limit=5)

        call_args = mock_search_all.call_args
        assert "specific query text" in call_args.args or "specific query text" in (
            call_args.kwargs.get("query", "")
        ), "search_all must receive the query string"

    @pytest.mark.asyncio
    async def test_search_papers_passes_limit_to_search_all(self):
        from academic_helper.tools.search import search_papers

        mock_search_all = AsyncMock(return_value=())
        with patch(
            "academic_helper.tools.search.search_all",
            new=mock_search_all,
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(),
        ):
            await search_papers("query", limit=7)

        call_args = mock_search_all.call_args
        assert 7 in call_args.args or call_args.kwargs.get("limit") == 7, (
            "search_all must receive the limit value"
        )

    @pytest.mark.asyncio
    async def test_search_papers_default_limit_is_10(self):
        """When limit is omitted, search_all should receive 10."""
        from academic_helper.tools.search import search_papers

        mock_search_all = AsyncMock(return_value=())
        with patch(
            "academic_helper.tools.search.search_all",
            new=mock_search_all,
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(),
        ):
            await search_papers("query")

        call_args = mock_search_all.call_args
        assert 10 in call_args.args or call_args.kwargs.get("limit") == 10, (
            "Default limit for search_papers must be 10"
        )

    @pytest.mark.asyncio
    async def test_search_papers_result_dict_has_required_fields(self):
        """Each dict in the result must have at least title, doi, abstract, authors."""
        from academic_helper.tools.search import search_papers

        paper = _make_paper(doi="10.1234/req", citation_count=1)
        result = _make_source_result("openalex", papers=(paper,))

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(result,)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(),),
        ):
            results = await search_papers("query", limit=5)

        assert len(results) >= 1
        rec = results[0]
        for field in ("title", "abstract", "doi", "authors"):
            assert field in rec, f"Result dict must contain '{field}'"

    @pytest.mark.asyncio
    async def test_search_papers_skips_errored_sources(self):
        """Papers from errored SourceResults must not appear in output."""
        from academic_helper.tools.search import search_papers

        good_paper = _make_paper(title="Good Paper", doi="10.9999/g")
        good_result = _make_source_result("openalex", papers=(good_paper,))
        bad_result = _make_source_result("crossref", error="timeout")

        with patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(good_result, bad_result)),
        ), patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(), MagicMock()),
        ):
            results = await search_papers("query", limit=10)

        titles = [r["title"] for r in results]
        assert "Good Paper" in titles
        # No stray entries from the errored source
        assert len(results) == 1


# ---------------------------------------------------------------------------
# deep_search
# ---------------------------------------------------------------------------


class TestDeepSearch:
    """deep_search(paper_doi, depth) finds citing/referenced papers via connectors."""

    @pytest.mark.asyncio
    async def test_deep_search_returns_list(self):
        from academic_helper.tools.search import deep_search

        with patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(),
        ), patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=()),
        ):
            result = await deep_search("10.1234/seed", depth=1)

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_deep_search_returns_list_of_dicts(self):
        from academic_helper.tools.search import deep_search

        paper = _make_paper(doi="10.0001/cited", title="Citing Paper")
        src_result = _make_source_result("openalex", papers=(paper,))

        with patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(),),
        ), patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(src_result,)),
        ):
            result = await deep_search("10.1234/seed", depth=1)

        assert all(isinstance(r, dict) for r in result)

    @pytest.mark.asyncio
    async def test_deep_search_uses_sources(self):
        """deep_search must call get_all_sources() to obtain connectors."""
        from academic_helper.tools.search import deep_search

        mock_get_sources = MagicMock(return_value=())
        with patch(
            "academic_helper.tools.search.get_all_sources",
            new=mock_get_sources,
        ), patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=()),
        ):
            await deep_search("10.1234/seed", depth=1)

        mock_get_sources.assert_called()

    @pytest.mark.asyncio
    async def test_deep_search_default_depth_is_1(self):
        """When depth is omitted, deep_search must still succeed."""
        from academic_helper.tools.search import deep_search

        with patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(),
        ), patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=()),
        ):
            result = await deep_search("10.1234/seed")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_deep_search_result_excludes_seed_doi(self):
        """Papers with the same DOI as the seed must not appear in results."""
        from academic_helper.tools.search import deep_search

        seed_doi = "10.1234/seed"
        seed_paper = _make_paper(doi=seed_doi, title="The Seed Paper")
        related = _make_paper(doi="10.5678/related", title="Related Paper")
        src_result = _make_source_result("openalex", papers=(seed_paper, related))

        with patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(),),
        ), patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(src_result,)),
        ):
            results = await deep_search(seed_doi, depth=1)

        dois = [r.get("doi") for r in results]
        assert seed_doi not in dois, "Seed paper DOI must be excluded from deep_search results"

    @pytest.mark.asyncio
    async def test_deep_search_result_dict_has_required_fields(self):
        from academic_helper.tools.search import deep_search

        paper = _make_paper(doi="10.0002/r")
        src_result = _make_source_result("semantic_scholar", papers=(paper,))

        with patch(
            "academic_helper.tools.search.get_all_sources",
            return_value=(MagicMock(),),
        ), patch(
            "academic_helper.tools.search.search_all",
            new=AsyncMock(return_value=(src_result,)),
        ):
            results = await deep_search("10.0000/seed", depth=1)

        if results:
            rec = results[0]
            for field in ("title", "abstract", "doi", "authors"):
                assert field in rec, f"deep_search result dict must contain '{field}'"


# ---------------------------------------------------------------------------
# expand_topics
# ---------------------------------------------------------------------------


class TestExpandTopics:
    """expand_topics(seed_topic, limit) returns related query strings."""

    @pytest.mark.asyncio
    async def test_expand_topics_returns_list(self):
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("nursing informatics")

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_expand_topics_returns_list_of_strings(self):
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("machine learning in healthcare")

        assert all(isinstance(item, str) for item in result), (
            "expand_topics must return a list of strings"
        )

    @pytest.mark.asyncio
    async def test_expand_topics_default_limit_is_5(self):
        """When limit is omitted, result length must not exceed 5."""
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("AI ethics")

        assert len(result) <= 5, "Default limit for expand_topics must be 5"

    @pytest.mark.asyncio
    async def test_expand_topics_respects_explicit_limit(self):
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("deep learning", limit=3)

        assert len(result) <= 3, "expand_topics must respect the provided limit"

    @pytest.mark.asyncio
    async def test_expand_topics_result_is_non_empty_for_valid_input(self):
        """A non-trivial topic should produce at least one related query."""
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("evidence-based nursing practice", limit=5)

        assert len(result) >= 1, (
            "expand_topics should return at least one related query for a valid topic"
        )

    @pytest.mark.asyncio
    async def test_expand_topics_queries_are_non_empty_strings(self):
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("telemedicine", limit=5)

        for item in result:
            assert len(item.strip()) > 0, "expand_topics must not return empty strings"

    @pytest.mark.asyncio
    async def test_expand_topics_does_not_return_duplicate_queries(self):
        from academic_helper.tools.search import expand_topics

        result = await expand_topics("clinical trials", limit=5)

        assert len(result) == len(set(result)), (
            "expand_topics must not return duplicate query strings"
        )


# ---------------------------------------------------------------------------
# extract_paper_from_pdf
# ---------------------------------------------------------------------------


class TestExtractPaperFromPdf:
    """extract_paper_from_pdf(pdf_path) extracts title, abstract, authors from PDF."""

    def _pdf_text(self) -> str:
        return (
            "Title: Machine Learning for Nursing Outcomes\n\n"
            "Authors: Wang, L.; Chen, M.; Liu, X.\n\n"
            "Abstract: This study investigates the use of machine learning "
            "algorithms to predict patient outcomes in nursing care settings.\n\n"
            "Introduction: Recent advances in..."
        )

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_returns_dict(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_result_has_title(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert "title" in result, "Extracted dict must contain 'title'"

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_result_has_abstract(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert "abstract" in result, "Extracted dict must contain 'abstract'"

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_result_has_authors(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert "authors" in result, "Extracted dict must contain 'authors'"

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_title_is_string(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert isinstance(result["title"], str)

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_abstract_is_string(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert isinstance(result["abstract"], str)

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_authors_is_list(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        assert isinstance(result["authors"], list), (
            "'authors' field must be a list"
        )

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_authors_are_strings(self):
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            return_value=self._pdf_text(),
        ):
            result = await extract_paper_from_pdf("/fake/path/paper.pdf")

        for author in result["authors"]:
            assert isinstance(author, str), "Each author must be a string"

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_file_not_found_raises(self):
        """Passing a non-existent path must raise a clear error."""
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch(
            "academic_helper.tools.pdf._read_pdf_text",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises((FileNotFoundError, ValueError)):
                await extract_paper_from_pdf("/nonexistent/path.pdf")

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_calls_read_pdf_text(self):
        """Must delegate actual PDF reading to an internal helper (_read_pdf_text)."""
        from academic_helper.tools.pdf import extract_paper_from_pdf

        mock_reader = MagicMock(return_value=self._pdf_text())
        with patch("academic_helper.tools.pdf._read_pdf_text", new=mock_reader):
            await extract_paper_from_pdf("/fake/paper.pdf")

        mock_reader.assert_called_once_with("/fake/paper.pdf")

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_empty_text_returns_partial_dict(self):
        """An empty PDF must still return a dict (with empty/None fields)."""
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch("academic_helper.tools.pdf._read_pdf_text", return_value=""):
            result = await extract_paper_from_pdf("/empty/paper.pdf")

        assert isinstance(result, dict)
        assert "title" in result
        assert "abstract" in result
        assert "authors" in result

    @pytest.mark.asyncio
    async def test_extract_paper_from_pdf_no_real_file_io(self):
        """With the mock in place, no actual file system reads should occur."""
        from academic_helper.tools.pdf import extract_paper_from_pdf

        with patch("academic_helper.tools.pdf._read_pdf_text", return_value="") as mock_io:
            await extract_paper_from_pdf("/fake/paper.pdf")

        # _read_pdf_text is the only sanctioned I/O path
        mock_io.assert_called_once()


# ---------------------------------------------------------------------------
# Integration-style: register() + real FastMCP
# ---------------------------------------------------------------------------


class TestRegisterIntegration:
    """Smoke-test that register() wires tools onto a live FastMCP instance."""

    def test_register_does_not_raise_with_fastmcp(self):
        from mcp.server.fastmcp import FastMCP

        from academic_helper.tools.search import register

        app = FastMCP("integration-test")
        register(app)  # must not raise

    def test_registered_tool_names_present(self):
        """After register(), the FastMCP instance must expose the three tool names."""
        from mcp.server.fastmcp import FastMCP

        from academic_helper.tools.search import register

        app = FastMCP("tool-name-test")
        register(app)

        # FastMCP stores tools keyed by name
        tool_names = set(app._tool_manager._tools.keys())
        assert "search_papers" in tool_names, "search_papers must be registered"
        assert "deep_search" in tool_names, "deep_search must be registered"
        assert "expand_topics" in tool_names, "expand_topics must be registered"
