"""search_all — concurrent helper that fans out across all academic sources."""

import asyncio
from collections.abc import Iterable

from academic_helper.models.source import SourceResult
from academic_helper.sources.base import AcademicSource


async def search_all(
    sources: Iterable[AcademicSource],
    query: str,
    limit: int,
) -> tuple[SourceResult, ...]:
    """Search all sources concurrently and return one SourceResult per source.

    Errors in individual sources are captured inside their SourceResult (fault
    isolation) and do not prevent other sources from returning results.

    Args:
        sources: Iterable of AcademicSource connectors to query.
        query: The search query string.
        limit: Maximum number of papers to request from each source.

    Returns:
        Tuple of SourceResult objects, one per source, in input order.
    """
    tasks = [source.search(query, limit) for source in sources]
    results = await asyncio.gather(*tasks)
    return tuple(results)
