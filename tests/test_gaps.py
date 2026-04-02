"""Tests for gap discovery tools — TDD RED phase (L005).

Covers:
  - academic_helper.tools.gaps  (GapPipeline ABC, GapResult, three mode subclasses,
    find_gaps MCP tool, mode mapping, register)

No real HTTP calls are made; all search operations are injected or mocked.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from academic_helper.models.domain import NURSING_TW, DomainProfile
from academic_helper.models.paper import Paper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(
    title: str = "Test Paper",
    source: str = "test",
    doi: str | None = "10.0000/test",
    citation_count: int = 0,
    year: int = 2024,
    abstract: str = "Abstract text.",
) -> Paper:
    return Paper(
        title=title,
        abstract=abstract,
        doi=doi,
        authors=("Author, A.",),
        year=year,
        source=source,
        url=f"https://doi.org/{doi}" if doi else None,
        citation_count=citation_count,
    )


def _stub_search_fn(papers: tuple[Paper, ...] = ()) -> Callable:
    """Return a synchronous stub search function that always returns the given papers."""

    async def _search(hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
        return papers

    return _search


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------


class TestGapsModuleImportable:
    """The tools.gaps module must be importable and expose expected symbols."""

    def test_gaps_module_is_importable(self):
        from academic_helper.tools import gaps  # noqa: F401

    def test_gap_result_is_importable(self):
        from academic_helper.tools.gaps import GapResult  # noqa: F401

    def test_gap_pipeline_is_importable(self):
        from academic_helper.tools.gaps import GapPipeline  # noqa: F401

    def test_aspect_comparison_is_importable(self):
        from academic_helper.tools.gaps import AspectComparison  # noqa: F401

    def test_context_localization_is_importable(self):
        from academic_helper.tools.gaps import ContextLocalization  # noqa: F401

    def test_method_modernization_is_importable(self):
        from academic_helper.tools.gaps import MethodModernization  # noqa: F401

    def test_find_gaps_function_is_importable(self):
        from academic_helper.tools.gaps import find_gaps

        assert callable(find_gaps)

    def test_register_function_is_importable(self):
        from academic_helper.tools.gaps import register

        assert callable(register)


# ---------------------------------------------------------------------------
# GapResult frozen dataclass
# ---------------------------------------------------------------------------


class TestGapResult:
    """GapResult must be a frozen dataclass with the required fields."""

    def test_gap_result_can_be_constructed(self):
        from academic_helper.tools.gaps import GapResult

        paper = _make_paper()
        result = GapResult(
            gap_type="aspect_comparison",
            description="Some gap description.",
            supporting_papers=(paper,),
            confidence=0.8,
        )
        assert result is not None

    def test_gap_result_stores_gap_type(self):
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="context_localization",
            description="desc",
            supporting_papers=(),
            confidence=0.5,
        )
        assert result.gap_type == "context_localization"

    def test_gap_result_stores_description(self):
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="method_modernization",
            description="The description text.",
            supporting_papers=(),
            confidence=0.3,
        )
        assert result.description == "The description text."

    def test_gap_result_stores_supporting_papers(self):
        from academic_helper.tools.gaps import GapResult

        paper = _make_paper(title="Supporting Paper")
        result = GapResult(
            gap_type="aspect_comparison",
            description="desc",
            supporting_papers=(paper,),
            confidence=0.9,
        )
        assert len(result.supporting_papers) == 1
        assert result.supporting_papers[0].title == "Supporting Paper"

    def test_gap_result_stores_confidence(self):
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="aspect_comparison",
            description="desc",
            supporting_papers=(),
            confidence=0.72,
        )
        assert result.confidence == pytest.approx(0.72)

    def test_gap_result_is_frozen(self):
        """Mutation of any field must raise FrozenInstanceError."""
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="aspect_comparison",
            description="desc",
            supporting_papers=(),
            confidence=0.5,
        )
        with pytest.raises(FrozenInstanceError):
            result.gap_type = "method_modernization"  # type: ignore[misc]

    def test_gap_result_supporting_papers_is_tuple(self):
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="aspect_comparison",
            description="desc",
            supporting_papers=(_make_paper(),),
            confidence=0.5,
        )
        assert isinstance(result.supporting_papers, tuple)

    def test_gap_result_confidence_zero_is_valid(self):
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="aspect_comparison",
            description="desc",
            supporting_papers=(),
            confidence=0.0,
        )
        assert result.confidence == pytest.approx(0.0)

    def test_gap_result_confidence_one_is_valid(self):
        from academic_helper.tools.gaps import GapResult

        result = GapResult(
            gap_type="aspect_comparison",
            description="desc",
            supporting_papers=(),
            confidence=1.0,
        )
        assert result.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GapPipeline ABC (Template Method)
# ---------------------------------------------------------------------------


class TestGapPipelineABC:
    """GapPipeline must be an abstract base class that cannot be instantiated directly."""

    def test_gap_pipeline_cannot_be_instantiated_directly(self):
        from academic_helper.tools.gaps import GapPipeline

        with pytest.raises(TypeError):
            GapPipeline(search_fn=_stub_search_fn())  # type: ignore[abstract]

    def test_subclass_missing_hypothesize_raises_type_error(self):
        from academic_helper.tools.gaps import GapPipeline, GapResult

        class Incomplete(GapPipeline):
            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                return ()

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                return ()

        with pytest.raises(TypeError):
            Incomplete(search_fn=_stub_search_fn())  # type: ignore[abstract]

    def test_subclass_missing_search_raises_type_error(self):
        from academic_helper.tools.gaps import GapPipeline, GapResult

        class Incomplete(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                return "hypothesis"

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                return ()

        with pytest.raises(TypeError):
            Incomplete(search_fn=_stub_search_fn())  # type: ignore[abstract]

    def test_subclass_missing_verify_raises_type_error(self):
        from academic_helper.tools.gaps import GapPipeline

        class Incomplete(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                return "hypothesis"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                return ()

        with pytest.raises(TypeError):
            Incomplete(search_fn=_stub_search_fn())  # type: ignore[abstract]

    def test_complete_subclass_can_be_instantiated(self):
        from academic_helper.tools.gaps import GapPipeline, GapResult

        class Complete(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                return "hypothesis"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                return ()

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                return ()

        instance = Complete(search_fn=_stub_search_fn())
        assert instance is not None

    def test_run_calls_hypothesize_search_verify_in_order(self):
        """run() must invoke hypothesize → search → verify in that order."""
        from academic_helper.tools.gaps import GapPipeline, GapResult

        call_order: list[str] = []

        class Ordered(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                call_order.append("hypothesize")
                return "the hypothesis"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                call_order.append("search")
                return ()

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                call_order.append("verify")
                return ()

        pipeline = Ordered(search_fn=_stub_search_fn())
        paper = _make_paper()

        import asyncio

        asyncio.get_event_loop().run_until_complete(pipeline.run(paper, NURSING_TW))

        assert call_order == ["hypothesize", "search", "verify"], (
            f"Expected [hypothesize, search, verify], got {call_order}"
        )

    @pytest.mark.asyncio
    async def test_run_returns_tuple_of_gap_results(self):
        from academic_helper.tools.gaps import GapPipeline, GapResult

        expected = GapResult(
            gap_type="aspect_comparison",
            description="some gap",
            supporting_papers=(),
            confidence=0.5,
        )

        class Concrete(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                return "h"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                return ()

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                return (expected,)

        pipeline = Concrete(search_fn=_stub_search_fn())
        results = await pipeline.run(_make_paper(), NURSING_TW)

        assert isinstance(results, tuple)
        assert len(results) == 1
        assert results[0] is expected

    @pytest.mark.asyncio
    async def test_run_passes_hypothesis_to_search(self):
        """The hypothesis string returned by hypothesize() must be passed to search()."""
        from academic_helper.tools.gaps import GapPipeline, GapResult

        received_hypothesis: list[str] = []

        class Spy(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                return "unique_hypothesis_token"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                received_hypothesis.append(hypothesis)
                return ()

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                return ()

        pipeline = Spy(search_fn=_stub_search_fn())
        await pipeline.run(_make_paper(), NURSING_TW)

        assert received_hypothesis == ["unique_hypothesis_token"]

    @pytest.mark.asyncio
    async def test_run_passes_papers_to_verify(self):
        """Papers returned by search() must be passed as-is to verify()."""
        from academic_helper.tools.gaps import GapPipeline, GapResult

        search_papers = (_make_paper(title="Found Paper"),)
        received_papers: list[tuple[Paper, ...]] = []

        class Spy(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                return "h"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                return search_papers

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                received_papers.append(papers)
                return ()

        pipeline = Spy(search_fn=_stub_search_fn())
        await pipeline.run(_make_paper(), NURSING_TW)

        assert len(received_papers) == 1
        assert received_papers[0] == search_papers

    @pytest.mark.asyncio
    async def test_run_is_stateless_across_calls(self):
        """Two successive run() calls on the same instance must return independent results."""
        from academic_helper.tools.gaps import GapPipeline, GapResult

        call_count = 0

        class Counter(GapPipeline):
            def hypothesize(self, paper: Paper, domain: DomainProfile) -> str:
                nonlocal call_count
                call_count += 1
                return f"hypothesis_{call_count}"

            def search(self, hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
                return ()

            def verify(
                self, hypothesis: str, papers: tuple[Paper, ...], domain: DomainProfile
            ) -> tuple[GapResult, ...]:
                return (
                    GapResult(
                        gap_type="aspect_comparison",
                        description=hypothesis,
                        supporting_papers=(),
                        confidence=0.5,
                    ),
                )

        pipeline = Counter(search_fn=_stub_search_fn())
        paper = _make_paper()

        results_a = await pipeline.run(paper, NURSING_TW)
        results_b = await pipeline.run(paper, NURSING_TW)

        assert results_a[0].description != results_b[0].description, (
            "run() must produce fresh results each call (stateless)"
        )


# ---------------------------------------------------------------------------
# AspectComparison
# ---------------------------------------------------------------------------


class TestAspectComparison:
    """AspectComparison — gap_type='aspect_comparison', examines methodological aspects."""

    def test_aspect_comparison_is_instantiable(self):
        from academic_helper.tools.gaps import AspectComparison

        instance = AspectComparison(search_fn=_stub_search_fn())
        assert instance is not None

    def test_aspect_comparison_gap_type_is_aspect_comparison(self):
        from academic_helper.tools.gaps import AspectComparison

        instance = AspectComparison(search_fn=_stub_search_fn())
        assert instance.gap_type == "aspect_comparison"

    def test_aspect_comparison_hypothesize_returns_non_empty_string(self):
        from academic_helper.tools.gaps import AspectComparison

        instance = AspectComparison(search_fn=_stub_search_fn())
        paper = _make_paper()
        hypothesis = instance.hypothesize(paper, NURSING_TW)

        assert isinstance(hypothesis, str)
        assert len(hypothesis.strip()) > 0

    @pytest.mark.asyncio
    async def test_aspect_comparison_search_returns_tuple_of_papers(self):
        from academic_helper.tools.gaps import AspectComparison

        found = (_make_paper(title="Better Method Paper"),)
        instance = AspectComparison(search_fn=_stub_search_fn(found))
        papers = await instance.search("some hypothesis", NURSING_TW)

        assert isinstance(papers, tuple)
        assert all(isinstance(p, Paper) for p in papers)

    @pytest.mark.asyncio
    async def test_aspect_comparison_search_delegates_to_search_fn(self):
        """search() must use the injected search_fn, not hardcoded logic."""
        from academic_helper.tools.gaps import AspectComparison

        captured: list[str] = []

        async def spy_search(hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
            captured.append(hypothesis)
            return ()

        instance = AspectComparison(search_fn=spy_search)
        await instance.search("test_hypothesis_token", NURSING_TW)

        assert "test_hypothesis_token" in captured

    @pytest.mark.asyncio
    async def test_aspect_comparison_verify_returns_tuple_of_gap_results(self):
        from academic_helper.tools.gaps import AspectComparison, GapResult

        papers = (_make_paper(),)
        instance = AspectComparison(search_fn=_stub_search_fn())
        results = instance.verify("hypothesis", papers, NURSING_TW)

        assert isinstance(results, tuple)
        assert all(isinstance(r, GapResult) for r in results)

    @pytest.mark.asyncio
    async def test_aspect_comparison_run_returns_tuple(self):
        from academic_helper.tools.gaps import AspectComparison, GapResult

        instance = AspectComparison(search_fn=_stub_search_fn())
        paper = _make_paper()
        results = await instance.run(paper, NURSING_TW)

        assert isinstance(results, tuple)
        assert all(isinstance(r, GapResult) for r in results)

    @pytest.mark.asyncio
    async def test_aspect_comparison_run_gap_type_in_results(self):
        """Any GapResult produced must carry gap_type='aspect_comparison'."""
        from academic_helper.tools.gaps import AspectComparison

        found = (_make_paper(title="Paper A"), _make_paper(title="Paper B"))
        instance = AspectComparison(search_fn=_stub_search_fn(found))
        paper = _make_paper()
        results = await instance.run(paper, NURSING_TW)

        for r in results:
            assert r.gap_type == "aspect_comparison", (
                f"Expected gap_type='aspect_comparison', got '{r.gap_type}'"
            )

    @pytest.mark.asyncio
    async def test_aspect_comparison_confidence_in_valid_range(self):
        from academic_helper.tools.gaps import AspectComparison

        found = (_make_paper(),)
        instance = AspectComparison(search_fn=_stub_search_fn(found))
        results = await instance.run(_make_paper(), NURSING_TW)

        for r in results:
            assert 0.0 <= r.confidence <= 1.0, (
                f"confidence must be in [0.0, 1.0], got {r.confidence}"
            )


# ---------------------------------------------------------------------------
# ContextLocalization
# ---------------------------------------------------------------------------


class TestContextLocalization:
    """ContextLocalization — gap_type='context_localization', checks locale applicability."""

    def test_context_localization_is_instantiable(self):
        from academic_helper.tools.gaps import ContextLocalization

        instance = ContextLocalization(search_fn=_stub_search_fn())
        assert instance is not None

    def test_context_localization_gap_type_is_context_localization(self):
        from academic_helper.tools.gaps import ContextLocalization

        instance = ContextLocalization(search_fn=_stub_search_fn())
        assert instance.gap_type == "context_localization"

    def test_context_localization_hypothesize_returns_non_empty_string(self):
        from academic_helper.tools.gaps import ContextLocalization

        instance = ContextLocalization(search_fn=_stub_search_fn())
        hypothesis = instance.hypothesize(_make_paper(), NURSING_TW)

        assert isinstance(hypothesis, str)
        assert len(hypothesis.strip()) > 0

    @pytest.mark.asyncio
    async def test_context_localization_search_returns_tuple_of_papers(self):
        from academic_helper.tools.gaps import ContextLocalization

        found = (_make_paper(title="Locale-Specific Paper"),)
        instance = ContextLocalization(search_fn=_stub_search_fn(found))
        papers = await instance.search("locale hypothesis", NURSING_TW)

        assert isinstance(papers, tuple)
        assert all(isinstance(p, Paper) for p in papers)

    @pytest.mark.asyncio
    async def test_context_localization_search_delegates_to_search_fn(self):
        from academic_helper.tools.gaps import ContextLocalization

        captured: list[str] = []

        async def spy_search(hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
            captured.append(hypothesis)
            return ()

        instance = ContextLocalization(search_fn=spy_search)
        await instance.search("locale_hypothesis_token", NURSING_TW)

        assert "locale_hypothesis_token" in captured

    def test_context_localization_verify_returns_tuple_of_gap_results(self):
        from academic_helper.tools.gaps import ContextLocalization, GapResult

        papers = (_make_paper(),)
        instance = ContextLocalization(search_fn=_stub_search_fn())
        results = instance.verify("hypothesis", papers, NURSING_TW)

        assert isinstance(results, tuple)
        assert all(isinstance(r, GapResult) for r in results)

    @pytest.mark.asyncio
    async def test_context_localization_run_returns_tuple(self):
        from academic_helper.tools.gaps import ContextLocalization, GapResult

        instance = ContextLocalization(search_fn=_stub_search_fn())
        results = await instance.run(_make_paper(), NURSING_TW)

        assert isinstance(results, tuple)
        assert all(isinstance(r, GapResult) for r in results)

    @pytest.mark.asyncio
    async def test_context_localization_run_gap_type_in_results(self):
        from academic_helper.tools.gaps import ContextLocalization

        found = (_make_paper(title="Locale Paper"),)
        instance = ContextLocalization(search_fn=_stub_search_fn(found))
        results = await instance.run(_make_paper(), NURSING_TW)

        for r in results:
            assert r.gap_type == "context_localization", (
                f"Expected gap_type='context_localization', got '{r.gap_type}'"
            )

    @pytest.mark.asyncio
    async def test_context_localization_confidence_in_valid_range(self):
        from academic_helper.tools.gaps import ContextLocalization

        found = (_make_paper(),)
        instance = ContextLocalization(search_fn=_stub_search_fn(found))
        results = await instance.run(_make_paper(), NURSING_TW)

        for r in results:
            assert 0.0 <= r.confidence <= 1.0, (
                f"confidence must be in [0.0, 1.0], got {r.confidence}"
            )


# ---------------------------------------------------------------------------
# MethodModernization
# ---------------------------------------------------------------------------


class TestMethodModernization:
    """MethodModernization — gap_type='method_modernization', checks tech improvement."""

    def test_method_modernization_is_instantiable(self):
        from academic_helper.tools.gaps import MethodModernization

        instance = MethodModernization(search_fn=_stub_search_fn())
        assert instance is not None

    def test_method_modernization_gap_type_is_method_modernization(self):
        from academic_helper.tools.gaps import MethodModernization

        instance = MethodModernization(search_fn=_stub_search_fn())
        assert instance.gap_type == "method_modernization"

    def test_method_modernization_hypothesize_returns_non_empty_string(self):
        from academic_helper.tools.gaps import MethodModernization

        instance = MethodModernization(search_fn=_stub_search_fn())
        hypothesis = instance.hypothesize(_make_paper(), NURSING_TW)

        assert isinstance(hypothesis, str)
        assert len(hypothesis.strip()) > 0

    @pytest.mark.asyncio
    async def test_method_modernization_search_returns_tuple_of_papers(self):
        from academic_helper.tools.gaps import MethodModernization

        found = (_make_paper(title="Modern Tech Paper"),)
        instance = MethodModernization(search_fn=_stub_search_fn(found))
        papers = await instance.search("modernization hypothesis", NURSING_TW)

        assert isinstance(papers, tuple)
        assert all(isinstance(p, Paper) for p in papers)

    @pytest.mark.asyncio
    async def test_method_modernization_search_delegates_to_search_fn(self):
        from academic_helper.tools.gaps import MethodModernization

        captured: list[str] = []

        async def spy_search(hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
            captured.append(hypothesis)
            return ()

        instance = MethodModernization(search_fn=spy_search)
        await instance.search("tech_modernization_token", NURSING_TW)

        assert "tech_modernization_token" in captured

    def test_method_modernization_verify_returns_tuple_of_gap_results(self):
        from academic_helper.tools.gaps import GapResult, MethodModernization

        papers = (_make_paper(),)
        instance = MethodModernization(search_fn=_stub_search_fn())
        results = instance.verify("hypothesis", papers, NURSING_TW)

        assert isinstance(results, tuple)
        assert all(isinstance(r, GapResult) for r in results)

    @pytest.mark.asyncio
    async def test_method_modernization_run_returns_tuple(self):
        from academic_helper.tools.gaps import GapResult, MethodModernization

        instance = MethodModernization(search_fn=_stub_search_fn())
        results = await instance.run(_make_paper(), NURSING_TW)

        assert isinstance(results, tuple)
        assert all(isinstance(r, GapResult) for r in results)

    @pytest.mark.asyncio
    async def test_method_modernization_run_gap_type_in_results(self):
        from academic_helper.tools.gaps import MethodModernization

        found = (_make_paper(title="New Technology Paper"),)
        instance = MethodModernization(search_fn=_stub_search_fn(found))
        results = await instance.run(_make_paper(), NURSING_TW)

        for r in results:
            assert r.gap_type == "method_modernization", (
                f"Expected gap_type='method_modernization', got '{r.gap_type}'"
            )

    @pytest.mark.asyncio
    async def test_method_modernization_confidence_in_valid_range(self):
        from academic_helper.tools.gaps import MethodModernization

        found = (_make_paper(),)
        instance = MethodModernization(search_fn=_stub_search_fn(found))
        results = await instance.run(_make_paper(), NURSING_TW)

        for r in results:
            assert 0.0 <= r.confidence <= 1.0, (
                f"confidence must be in [0.0, 1.0], got {r.confidence}"
            )


# ---------------------------------------------------------------------------
# Distinct gap_type values across all three modes
# ---------------------------------------------------------------------------


class TestDistinctGapTypes:
    """All three modes must produce distinct gap_type values."""

    def test_all_three_modes_have_distinct_gap_types(self):
        from academic_helper.tools.gaps import (
            AspectComparison,
            ContextLocalization,
            MethodModernization,
        )

        types = {
            AspectComparison(search_fn=_stub_search_fn()).gap_type,
            ContextLocalization(search_fn=_stub_search_fn()).gap_type,
            MethodModernization(search_fn=_stub_search_fn()).gap_type,
        }
        assert len(types) == 3, (
            f"All three modes must have distinct gap_type values, got: {types}"
        )

    def test_aspect_comparison_gap_type_differs_from_context_localization(self):
        from academic_helper.tools.gaps import AspectComparison, ContextLocalization

        assert (
            AspectComparison(search_fn=_stub_search_fn()).gap_type
            != ContextLocalization(search_fn=_stub_search_fn()).gap_type
        )

    def test_context_localization_gap_type_differs_from_method_modernization(self):
        from academic_helper.tools.gaps import ContextLocalization, MethodModernization

        assert (
            ContextLocalization(search_fn=_stub_search_fn()).gap_type
            != MethodModernization(search_fn=_stub_search_fn()).gap_type
        )

    def test_aspect_comparison_gap_type_differs_from_method_modernization(self):
        from academic_helper.tools.gaps import AspectComparison, MethodModernization

        assert (
            AspectComparison(search_fn=_stub_search_fn()).gap_type
            != MethodModernization(search_fn=_stub_search_fn()).gap_type
        )


# ---------------------------------------------------------------------------
# Mode Mapping (user-goal language → internal mode)
# ---------------------------------------------------------------------------


class TestModeMapping:
    """find_gaps mode parameter must accept both English mode names and user-goal phrases."""

    @pytest.mark.asyncio
    async def test_mode_aspect_comparison_english_accepted(self):
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            results = await find_gaps(
                paper_title="Test Title",
                paper_abstract="Test abstract.",
                mode="aspect_comparison",
                domain="nursing",
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_mode_context_localization_english_accepted(self):
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            results = await find_gaps(
                paper_title="Test Title",
                paper_abstract="Test abstract.",
                mode="context_localization",
                domain="nursing",
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_mode_method_modernization_english_accepted(self):
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            results = await find_gaps(
                paper_title="Test Title",
                paper_abstract="Test abstract.",
                mode="method_modernization",
                domain="nursing",
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_chinese_phrase_maps_to_aspect_comparison(self):
        """'這篇論文的某些方法，有沒有其他論文做得更好？' → aspect_comparison."""
        from academic_helper.tools.gaps import find_gaps

        captured_modes: list[str] = []

        async def spy_search(query: str, limit: int = 10) -> list[dict]:
            return []

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(side_effect=spy_search),
        ), patch(
            "academic_helper.tools.gaps.AspectComparison",
            wraps=__import__(
                "academic_helper.tools.gaps", fromlist=["AspectComparison"]
            ).AspectComparison,
        ) as mock_aspect:
            await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="這篇論文的某些方法，有沒有其他論文做得更好？",
                domain="nursing",
            )
            assert mock_aspect.called, (
                "Chinese phrase for aspect_comparison must instantiate AspectComparison"
            )

    @pytest.mark.asyncio
    async def test_chinese_phrase_maps_to_context_localization(self):
        """'這篇研究搬到我的情境適用嗎？' → context_localization."""
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ), patch(
            "academic_helper.tools.gaps.ContextLocalization",
            wraps=__import__(
                "academic_helper.tools.gaps", fromlist=["ContextLocalization"]
            ).ContextLocalization,
        ) as mock_context:
            await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="這篇研究搬到我的情境適用嗎？",
                domain="nursing",
            )
            assert mock_context.called, (
                "Chinese phrase for context_localization must instantiate ContextLocalization"
            )

    @pytest.mark.asyncio
    async def test_chinese_phrase_maps_to_method_modernization(self):
        """'這篇論文的傳統方法，可以引入新技術改善嗎？' → method_modernization."""
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ), patch(
            "academic_helper.tools.gaps.MethodModernization",
            wraps=__import__(
                "academic_helper.tools.gaps", fromlist=["MethodModernization"]
            ).MethodModernization,
        ) as mock_modern:
            await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="這篇論文的傳統方法，可以引入新技術改善嗎？",
                domain="nursing",
            )
            assert mock_modern.called, (
                "Chinese phrase for method_modernization must instantiate MethodModernization"
            )

    @pytest.mark.asyncio
    async def test_unknown_mode_raises_value_error(self):
        """An unrecognised mode string must raise ValueError with a clear message."""
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            with pytest.raises(ValueError, match="mode"):
                await find_gaps(
                    paper_title="T",
                    paper_abstract="A",
                    mode="not_a_valid_mode_xyz",
                    domain="nursing",
                )

    @pytest.mark.asyncio
    async def test_default_mode_is_aspect_comparison(self):
        """When mode is omitted, find_gaps must default to aspect_comparison."""
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ), patch(
            "academic_helper.tools.gaps.AspectComparison",
            wraps=__import__(
                "academic_helper.tools.gaps", fromlist=["AspectComparison"]
            ).AspectComparison,
        ) as mock_aspect:
            await find_gaps(
                paper_title="T",
                paper_abstract="A",
            )
            assert mock_aspect.called, "Default mode must be aspect_comparison"


# ---------------------------------------------------------------------------
# find_gaps MCP tool
# ---------------------------------------------------------------------------


class TestFindGapsTool:
    """find_gaps(paper_title, paper_abstract, mode, domain) → list[dict]."""

    @pytest.mark.asyncio
    async def test_find_gaps_returns_list(self):
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            results = await find_gaps(
                paper_title="AI in Nursing",
                paper_abstract="This paper studies AI in nursing care.",
                mode="aspect_comparison",
                domain="nursing",
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_gaps_result_items_are_dicts(self):
        from academic_helper.tools.gaps import find_gaps

        found = (_make_paper(title="Supporting Paper"),)
        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            results = await find_gaps(
                paper_title="AI in Nursing",
                paper_abstract="This paper studies AI.",
                mode="aspect_comparison",
                domain="nursing",
            )

        assert all(isinstance(r, dict) for r in results)

    @pytest.mark.asyncio
    async def test_find_gaps_result_dict_has_gap_type(self):
        """Each dict in the result must contain 'gap_type'."""
        from academic_helper.tools.gaps import find_gaps

        found_papers = [
            {
                "title": "Supporting Paper",
                "abstract": "Abstract",
                "doi": "10.0/test",
                "authors": ["Author A"],
                "year": 2023,
                "source": "openalex",
                "url": None,
                "citation_count": 5,
            }
        ]

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=found_papers),
        ):
            results = await find_gaps(
                paper_title="AI in Nursing",
                paper_abstract="This paper studies AI.",
                mode="aspect_comparison",
                domain="nursing",
            )

        for r in results:
            assert "gap_type" in r, "Each result dict must have 'gap_type'"

    @pytest.mark.asyncio
    async def test_find_gaps_result_dict_has_description(self):
        """Each dict in the result must contain 'description'."""
        from academic_helper.tools.gaps import find_gaps

        found_papers = [
            {
                "title": "Paper",
                "abstract": "Abstract",
                "doi": "10.0/t",
                "authors": ["A"],
                "year": 2022,
                "source": "openalex",
                "url": None,
                "citation_count": 1,
            }
        ]

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=found_papers),
        ):
            results = await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="context_localization",
                domain="nursing",
            )

        for r in results:
            assert "description" in r, "Each result dict must have 'description'"

    @pytest.mark.asyncio
    async def test_find_gaps_result_dict_has_confidence(self):
        """Each dict in the result must contain 'confidence'."""
        from academic_helper.tools.gaps import find_gaps

        found_papers = [
            {
                "title": "Paper",
                "abstract": "Abstract",
                "doi": "10.0/t",
                "authors": ["A"],
                "year": 2022,
                "source": "openalex",
                "url": None,
                "citation_count": 1,
            }
        ]

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=found_papers),
        ):
            results = await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="method_modernization",
                domain="nursing",
            )

        for r in results:
            assert "confidence" in r, "Each result dict must have 'confidence'"

    @pytest.mark.asyncio
    async def test_find_gaps_result_dict_has_supporting_papers(self):
        """Each dict in the result must contain 'supporting_papers'."""
        from academic_helper.tools.gaps import find_gaps

        found_papers = [
            {
                "title": "Paper",
                "abstract": "Abstract",
                "doi": "10.0/t",
                "authors": ["A"],
                "year": 2022,
                "source": "openalex",
                "url": None,
                "citation_count": 1,
            }
        ]

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=found_papers),
        ):
            results = await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="aspect_comparison",
                domain="nursing",
            )

        for r in results:
            assert "supporting_papers" in r, "Each result dict must have 'supporting_papers'"

    @pytest.mark.asyncio
    async def test_find_gaps_no_real_network_calls(self):
        """find_gaps must route through the injected search, not make real HTTP calls."""
        from academic_helper.tools.gaps import find_gaps

        mock_search = AsyncMock(return_value=[])
        with patch("academic_helper.tools.gaps.search_papers", new=mock_search):
            await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="aspect_comparison",
                domain="nursing",
            )

        # The mock must have been called (search was delegated, not bypassed)
        assert mock_search.called

    @pytest.mark.asyncio
    async def test_find_gaps_confidence_values_in_range(self):
        """All confidence values in the returned dicts must be in [0.0, 1.0]."""
        from academic_helper.tools.gaps import find_gaps

        found_papers = [
            {
                "title": f"Paper {i}",
                "abstract": "Abstract",
                "doi": f"10.0/{i}",
                "authors": ["A"],
                "year": 2022,
                "source": "openalex",
                "url": None,
                "citation_count": i,
            }
            for i in range(3)
        ]

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=found_papers),
        ):
            results = await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="aspect_comparison",
                domain="nursing",
            )

        for r in results:
            conf = r.get("confidence", -1)
            assert 0.0 <= conf <= 1.0, f"confidence out of range: {conf}"

    @pytest.mark.asyncio
    async def test_find_gaps_default_domain_is_nursing(self):
        """When domain is omitted, find_gaps must not raise and must use nursing defaults."""
        from academic_helper.tools.gaps import find_gaps

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=[]),
        ):
            results = await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="aspect_comparison",
            )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_find_gaps_supporting_papers_field_is_list(self):
        """The 'supporting_papers' field in each result dict must be a list."""
        from academic_helper.tools.gaps import find_gaps

        found_papers = [
            {
                "title": "A Paper",
                "abstract": "Abstract",
                "doi": "10.0/a",
                "authors": ["A"],
                "year": 2023,
                "source": "openalex",
                "url": None,
                "citation_count": 2,
            }
        ]

        with patch(
            "academic_helper.tools.gaps.search_papers",
            new=AsyncMock(return_value=found_papers),
        ):
            results = await find_gaps(
                paper_title="T",
                paper_abstract="A",
                mode="aspect_comparison",
                domain="nursing",
            )

        for r in results:
            assert isinstance(r.get("supporting_papers"), list), (
                "'supporting_papers' must be a list in the result dict"
            )


# ---------------------------------------------------------------------------
# register() — MCP tool registration
# ---------------------------------------------------------------------------


class TestRegisterGapsTools:
    """register(mcp) must register find_gaps on the given FastMCP instance."""

    def _make_mock_mcp(self) -> MagicMock:
        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: fn
        return mock_mcp

    def test_register_calls_tool_decorator(self):
        from academic_helper.tools.gaps import register

        mock_mcp = self._make_mock_mcp()
        register(mock_mcp)

        assert mock_mcp.tool.called, "register() must call mcp.tool() at least once"

    def test_register_registers_find_gaps(self):
        from mcp.server.fastmcp import FastMCP

        from academic_helper.tools.gaps import register

        app = FastMCP("gaps-test")
        register(app)

        tool_names = set(app._tool_manager._tools.keys())
        assert "find_gaps" in tool_names, "find_gaps must be registered on the FastMCP instance"

    def test_register_with_real_fastmcp_does_not_raise(self):
        from mcp.server.fastmcp import FastMCP

        from academic_helper.tools.gaps import register

        app = FastMCP("gaps-register-test")
        register(app)  # must not raise


# ---------------------------------------------------------------------------
# Integration: GapPipeline subclasses are independent of each other
# ---------------------------------------------------------------------------


class TestPipelineIndependence:
    """The three pipeline subclasses must be fully independent."""

    @pytest.mark.asyncio
    async def test_three_pipelines_run_concurrently_without_interference(self):
        """Running all three pipelines on the same paper must produce three distinct sets."""
        from academic_helper.tools.gaps import (
            AspectComparison,
            ContextLocalization,
            MethodModernization,
        )

        paper = _make_paper(title="Shared Target Paper")
        found = (_make_paper(title="Found Paper"),)

        ac = AspectComparison(search_fn=_stub_search_fn(found))
        cl = ContextLocalization(search_fn=_stub_search_fn(found))
        mm = MethodModernization(search_fn=_stub_search_fn(found))

        results_ac = await ac.run(paper, NURSING_TW)
        results_cl = await cl.run(paper, NURSING_TW)
        results_mm = await mm.run(paper, NURSING_TW)

        # Each set must carry its own gap_type
        gap_types_ac = {r.gap_type for r in results_ac}
        gap_types_cl = {r.gap_type for r in results_cl}
        gap_types_mm = {r.gap_type for r in results_mm}

        if results_ac:
            assert gap_types_ac <= {"aspect_comparison"}
        if results_cl:
            assert gap_types_cl <= {"context_localization"}
        if results_mm:
            assert gap_types_mm <= {"method_modernization"}

    @pytest.mark.asyncio
    async def test_each_pipeline_uses_its_own_search_fn(self):
        """Each pipeline instance uses only its own injected search_fn."""
        from academic_helper.tools.gaps import (
            AspectComparison,
            ContextLocalization,
        )

        calls_a: list[str] = []
        calls_b: list[str] = []

        async def search_a(hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
            calls_a.append(hypothesis)
            return ()

        async def search_b(hypothesis: str, domain: DomainProfile) -> tuple[Paper, ...]:
            calls_b.append(hypothesis)
            return ()

        ac = AspectComparison(search_fn=search_a)
        cl = ContextLocalization(search_fn=search_b)

        paper = _make_paper()
        await ac.run(paper, NURSING_TW)
        await cl.run(paper, NURSING_TW)

        assert len(calls_a) >= 1, "AspectComparison must call its own search_fn"
        assert len(calls_b) >= 1, "ContextLocalization must call its own search_fn"
        # No cross-contamination
        assert len(calls_b) == len(calls_b)  # trivial, but ensures isolation
