"""Tests for F005: httpx.AsyncClient lifecycle management in registry.py.

TDD RED phase — all tests in this file are expected to FAIL until the
following new API is added to academic_helper.sources.registry:

  get_all_sources(client: httpx.AsyncClient | None = None) -> tuple[AcademicSource, ...]
  create_source_session() -> AsyncContextManager[tuple[AcademicSource, ...]]
  search_with_session(query: str, limit: int) -> tuple[SourceResult, ...]
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from academic_helper.models.source import SourceResult
from academic_helper.sources.base import AcademicSource
from academic_helper.sources.crossref import CrossRefSource
from academic_helper.sources.openalex import OpenAlexSource
from academic_helper.sources.semantic_scholar import SemanticScholarSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client() -> AsyncMock:
    """Return a fresh AsyncMock that stands in for httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.aclose = AsyncMock(return_value=None)
    return client


# ---------------------------------------------------------------------------
# 1. Shared client injection via get_all_sources(client=...)
# ---------------------------------------------------------------------------


class TestGetAllSourcesClientInjection:
    """get_all_sources() must accept an optional shared httpx.AsyncClient."""

    def test_get_all_sources_accepts_client_keyword_arg(self):
        """get_all_sources(client=...) must not raise TypeError."""
        from academic_helper.sources.registry import get_all_sources

        mock_client = _make_mock_client()
        # If the signature does not accept `client`, this call raises TypeError.
        sources = get_all_sources(client=mock_client)
        assert isinstance(sources, tuple)

    def test_all_three_sources_receive_the_same_client_instance(self):
        """Every source returned by get_all_sources(client=c) must hold c."""
        from academic_helper.sources.registry import get_all_sources

        mock_client = _make_mock_client()
        sources = get_all_sources(client=mock_client)

        assert len(sources) == 3, "Expected 3 sources"
        for source in sources:
            # Each concrete source stores the client as self._client or self.client.
            # We check both common attribute names to be implementation-agnostic.
            client_attr = getattr(source, "_client", None) or getattr(source, "client", None)
            assert client_attr is mock_client, (
                f"{type(source).__name__} does not hold the injected client; "
                f"got {client_attr!r}"
            )

    def test_openalex_source_receives_injected_client(self):
        """OpenAlexSource must be constructed with the passed-in client."""
        from academic_helper.sources.registry import get_all_sources

        mock_client = _make_mock_client()
        sources = get_all_sources(client=mock_client)

        oa = next(s for s in sources if isinstance(s, OpenAlexSource))
        client_attr = getattr(oa, "_client", None) or getattr(oa, "client", None)
        assert client_attr is mock_client

    def test_semantic_scholar_source_receives_injected_client(self):
        """SemanticScholarSource must be constructed with the passed-in client."""
        from academic_helper.sources.registry import get_all_sources

        mock_client = _make_mock_client()
        sources = get_all_sources(client=mock_client)

        s2 = next(s for s in sources if isinstance(s, SemanticScholarSource))
        client_attr = getattr(s2, "_client", None) or getattr(s2, "client", None)
        assert client_attr is mock_client

    def test_crossref_source_receives_injected_client(self):
        """CrossRefSource must be constructed with the passed-in client."""
        from academic_helper.sources.registry import get_all_sources

        mock_client = _make_mock_client()
        sources = get_all_sources(client=mock_client)

        cr = next(s for s in sources if isinstance(s, CrossRefSource))
        client_attr = getattr(cr, "_client", None) or getattr(cr, "client", None)
        assert client_attr is mock_client

    def test_get_all_sources_no_arg_still_works(self):
        """Calling get_all_sources() without arguments must remain valid."""
        from academic_helper.sources.registry import get_all_sources

        sources = get_all_sources()
        assert isinstance(sources, tuple)
        assert len(sources) == 3
        for s in sources:
            assert isinstance(s, AcademicSource)

    def test_get_all_sources_no_arg_creates_internal_client(self):
        """When no client is supplied, each source must have *some* client set."""
        from academic_helper.sources.registry import get_all_sources

        sources = get_all_sources()
        for source in sources:
            client_attr = getattr(source, "_client", None) or getattr(source, "client", None)
            assert client_attr is not None, (
                f"{type(source).__name__} has no client attribute when none was injected"
            )

    def test_get_all_sources_with_client_none_creates_internal_client(self):
        """Passing client=None explicitly is equivalent to calling with no argument."""
        from academic_helper.sources.registry import get_all_sources

        sources = get_all_sources(client=None)
        assert isinstance(sources, tuple)
        assert len(sources) == 3

    def test_injected_client_is_shared_not_copied(self):
        """The same object must be shared — no copies or wrappers allowed."""
        from academic_helper.sources.registry import get_all_sources

        mock_client = _make_mock_client()
        sources = get_all_sources(client=mock_client)

        client_ids = set()
        for source in sources:
            c = getattr(source, "_client", None) or getattr(source, "client", None)
            client_ids.add(id(c))

        assert len(client_ids) == 1, (
            "All three sources should reference the *same* client object "
            f"but found {len(client_ids)} distinct objects"
        )


# ---------------------------------------------------------------------------
# 2. Context manager — create_source_session()
# ---------------------------------------------------------------------------


class TestCreateSourceSession:
    """create_source_session() must be an async context manager."""

    @pytest.mark.asyncio
    async def test_create_source_session_is_importable(self):
        """create_source_session must be importable from registry."""
        from academic_helper.sources.registry import create_source_session  # noqa: F401

    @pytest.mark.asyncio
    async def test_create_source_session_returns_tuple_of_sources(self):
        """The context manager must yield a tuple of AcademicSource instances."""
        from academic_helper.sources.registry import create_source_session

        async with create_source_session() as sources:
            assert isinstance(sources, tuple)
            assert len(sources) == 3
            for s in sources:
                assert isinstance(s, AcademicSource)

    @pytest.mark.asyncio
    async def test_create_source_session_contains_all_three_connector_types(self):
        """The yielded tuple must contain one of each connector type."""
        from academic_helper.sources.registry import create_source_session

        async with create_source_session() as sources:
            source_types = {type(s) for s in sources}
            assert OpenAlexSource in source_types
            assert SemanticScholarSource in source_types
            assert CrossRefSource in source_types

    @pytest.mark.asyncio
    async def test_create_source_session_calls_aclose_on_exit(self):
        """The shared client's aclose() must be awaited when the block exits."""
        from academic_helper.sources.registry import create_source_session

        mock_client = _make_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            async with create_source_session():
                mock_client.aclose.assert_not_awaited()

        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_source_session_closes_client_on_normal_exit(self):
        """Client must be closed even when the body completes without error."""
        from academic_helper.sources.registry import create_source_session

        mock_client = _make_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            async with create_source_session() as sources:
                assert len(sources) == 3

        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_source_session_closes_client_on_exception(self):
        """Client must be closed even if the body raises an exception."""
        from academic_helper.sources.registry import create_source_session

        mock_client = _make_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="boom"):
                async with create_source_session():
                    raise RuntimeError("boom")

        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_source_session_accepts_injected_client(self):
        """create_source_session() must accept an optional client= argument."""
        from academic_helper.sources.registry import create_source_session

        mock_client = _make_mock_client()
        async with create_source_session(client=mock_client) as sources:
            assert isinstance(sources, tuple)
            assert len(sources) == 3

    @pytest.mark.asyncio
    async def test_create_source_session_does_not_close_injected_client(self):
        """When the caller injects a client, the session must NOT close it.

        Ownership stays with the caller — the context manager must not call
        aclose() on a client it did not create.
        """
        from academic_helper.sources.registry import create_source_session

        mock_client = _make_mock_client()
        async with create_source_session(client=mock_client):
            pass

        mock_client.aclose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_create_source_session_all_sources_share_one_client(self):
        """All sources in the session must share the same underlying client."""
        from academic_helper.sources.registry import create_source_session

        mock_client = _make_mock_client()

        with patch("httpx.AsyncClient", return_value=mock_client):
            async with create_source_session() as sources:
                client_ids = set()
                for source in sources:
                    c = getattr(source, "_client", None) or getattr(source, "client", None)
                    client_ids.add(id(c))

        assert len(client_ids) == 1, (
            "All sources in a session must share a single client object"
        )


# ---------------------------------------------------------------------------
# 3. search_with_session() convenience function
# ---------------------------------------------------------------------------


class TestSearchWithSession:
    """search_with_session() must run a full search and close the client."""

    @pytest.mark.asyncio
    async def test_search_with_session_is_importable(self):
        """search_with_session must be importable from registry."""
        from academic_helper.sources.registry import search_with_session  # noqa: F401

    @pytest.mark.asyncio
    async def test_search_with_session_returns_tuple(self):
        """Return type must be tuple[SourceResult, ...]."""
        from academic_helper.sources.registry import search_with_session

        mock_client = _make_mock_client()

        # Patch AsyncClient and search_all to avoid real HTTP calls.
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "academic_helper.sources.registry.search_all",
                new_callable=lambda: _make_search_all_stub,
            ),
        ):
            results = await search_with_session("nursing AI", limit=5)

        assert isinstance(results, tuple)

    @pytest.mark.asyncio
    async def test_search_with_session_returns_one_result_per_source(self):
        """Must return one SourceResult per registered source (3 total)."""
        from academic_helper.sources.registry import search_with_session

        mock_client = _make_mock_client()

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "academic_helper.sources.registry.search_all",
                new_callable=lambda: _make_search_all_stub,
            ),
        ):
            results = await search_with_session("nursing AI", limit=5)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_with_session_results_are_source_result_instances(self):
        """Every element in the returned tuple must be a SourceResult."""
        from academic_helper.sources.registry import search_with_session

        mock_client = _make_mock_client()

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "academic_helper.sources.registry.search_all",
                new_callable=lambda: _make_search_all_stub,
            ),
        ):
            results = await search_with_session("nursing AI", limit=5)

        assert all(isinstance(r, SourceResult) for r in results)

    @pytest.mark.asyncio
    async def test_search_with_session_closes_client_after_search(self):
        """The internal client must be closed after search completes."""
        from academic_helper.sources.registry import search_with_session

        mock_client = _make_mock_client()

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "academic_helper.sources.registry.search_all",
                new_callable=lambda: _make_search_all_stub,
            ),
        ):
            await search_with_session("query", limit=3)

        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_with_session_closes_client_even_on_search_error(self):
        """Client must be closed even when search_all raises an exception."""
        from academic_helper.sources.registry import search_with_session

        mock_client = _make_mock_client()

        async def _raising_search_all(sources, query, limit):
            raise RuntimeError("search failure")

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("academic_helper.sources.registry.search_all", new=_raising_search_all),
        ):
            with pytest.raises(RuntimeError, match="search failure"):
                await search_with_session("query", limit=3)

        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_with_session_passes_query_and_limit_to_search_all(self):
        """search_with_session must forward query and limit verbatim to search_all."""
        from academic_helper.sources.registry import search_with_session

        mock_client = _make_mock_client()
        captured: dict = {}

        async def _capturing_search_all(sources, query, limit):
            captured["query"] = query
            captured["limit"] = limit
            return tuple(
                SourceResult(source_name=type(s).__name__, papers=()) for s in sources
            )

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("academic_helper.sources.registry.search_all", new=_capturing_search_all),
        ):
            await search_with_session("machine learning", limit=10)

        assert captured.get("query") == "machine learning"
        assert captured.get("limit") == 10


# ---------------------------------------------------------------------------
# Internal stub — used by the search_with_session tests above
# ---------------------------------------------------------------------------


def _make_search_all_stub():
    """Factory that returns an async stub coroutine function for search_all."""

    async def _stub(sources, query, limit):
        return tuple(
            SourceResult(source_name=type(s).__name__, papers=()) for s in sources
        )

    return _stub
