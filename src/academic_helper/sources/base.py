"""AcademicSource — abstract base class for all academic API connectors."""

from abc import ABC, abstractmethod

from academic_helper.models.paper import Paper
from academic_helper.models.source import SourceResult


class AcademicSource(ABC):
    """Abstract base class defining the interface for academic search sources."""

    @abstractmethod
    async def search(self, query: str, limit: int) -> SourceResult:
        """Search for papers matching the query and return a SourceResult.

        Args:
            query: The search query string.
            limit: Maximum number of papers to return.

        Returns:
            SourceResult containing papers or an error message.
        """

    @abstractmethod
    async def get_paper(self, doi: str) -> Paper | None:
        """Fetch a single paper by DOI.

        Args:
            doi: The DOI string identifying the paper.

        Returns:
            Paper if found, None on 404 or HTTP error.
        """

    @abstractmethod
    async def get_citations(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers that cite the given DOI.

        Args:
            doi: The DOI of the paper whose citations to retrieve.
            limit: Maximum number of citing papers to return.

        Returns:
            SourceResult containing citing papers or an error message.
        """

    @abstractmethod
    async def get_references(self, doi: str, limit: int = 20) -> SourceResult:
        """Fetch papers referenced by the given DOI.

        Args:
            doi: The DOI of the paper whose references to retrieve.
            limit: Maximum number of referenced papers to return.

        Returns:
            SourceResult containing referenced papers or an error message.
        """
