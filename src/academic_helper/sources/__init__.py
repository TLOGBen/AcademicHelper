"""Academic source connectors package."""

from academic_helper.sources.base import AcademicSource
from academic_helper.sources.crossref import CrossRefSource
from academic_helper.sources.openalex import OpenAlexSource
from academic_helper.sources.registry import get_all_sources
from academic_helper.sources.search import search_all
from academic_helper.sources.semantic_scholar import SemanticScholarSource

__all__ = [
    "AcademicSource",
    "CrossRefSource",
    "OpenAlexSource",
    "SemanticScholarSource",
    "get_all_sources",
    "search_all",
]
