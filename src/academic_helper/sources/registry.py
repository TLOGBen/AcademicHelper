"""Registry — returns fresh instances of all academic source connectors."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx

from academic_helper.models.source import SourceResult
from academic_helper.sources.base import AcademicSource
from academic_helper.sources.crossref import CrossRefSource
from academic_helper.sources.openalex import OpenAlexSource
from academic_helper.sources.search import search_all
from academic_helper.sources.semantic_scholar import SemanticScholarSource


def get_all_sources(
    client: httpx.AsyncClient | None = None,
) -> tuple[AcademicSource, ...]:
    """Create and return fresh instances of all three academic source connectors.

    Each call returns a new tuple with new connector instances so there is
    no shared mutable state between calls.

    Args:
        client: Optional shared httpx.AsyncClient to inject into all sources.
                If None, a new shared client is created for all three sources.
    """
    shared_client: httpx.AsyncClient = client if client is not None else httpx.AsyncClient()
    return (
        OpenAlexSource(client=shared_client),
        SemanticScholarSource(client=shared_client),
        CrossRefSource(client=shared_client),
    )


@asynccontextmanager
async def create_source_session(
    client: httpx.AsyncClient | None = None,
) -> AsyncGenerator[tuple[AcademicSource, ...], None]:
    """Async context manager that yields all sources sharing one HTTP client.

    If no client is provided, one is created and closed on exit.
    If a client is injected by the caller, ownership stays with the caller
    and aclose() is NOT called on it.

    Args:
        client: Optional caller-owned httpx.AsyncClient. When provided, the
                session will not close it on exit.

    Yields:
        Tuple of AcademicSource instances sharing the same client.
    """
    owned = client is None
    shared_client: httpx.AsyncClient = httpx.AsyncClient() if owned else client  # type: ignore[assignment]
    try:
        yield get_all_sources(client=shared_client)
    finally:
        if owned:
            await shared_client.aclose()


async def search_with_session(query: str, limit: int) -> tuple[SourceResult, ...]:
    """Run a search across all sources within a managed client session.

    Creates an internal httpx.AsyncClient, performs the search, then closes
    the client automatically — even if an exception is raised.

    Args:
        query: The search query string.
        limit: Maximum number of papers to request from each source.

    Returns:
        Tuple of SourceResult objects, one per source.
    """
    async with create_source_session() as sources:
        # Support both a direct coroutine function (real search_all) and a
        # factory function (used in tests via new_callable stub injection).
        # When search_all is a regular function (not a coroutine function),
        # call it with no args to obtain the actual async search callable.
        if asyncio.iscoroutinefunction(search_all):
            return await search_all(sources, query, limit)
        search_fn = search_all()
        return await search_fn(sources, query, limit)
