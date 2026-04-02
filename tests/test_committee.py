"""Tests for committee review tools and models."""

from __future__ import annotations

import inspect

import pytest

from academic_helper.models.committee import ReviewContext


# ---------------------------------------------------------------------------
# 1. ReviewContext frozen dataclass
# ---------------------------------------------------------------------------


class TestReviewContext:
    """Tests for ReviewContext frozen dataclass."""

    def test_create_review_context(self):
        """ReviewContext can be created with all required fields."""
        ctx = ReviewContext(
            papers=({"title": "Paper 1", "abstract": "Abstract 1"},),
            committee_members=({"name": "test", "display_name": "Test", "focus": "Testing"},),
            rubric={"test": ("dim1", "dim2")},
            domain="nursing",
        )
        assert ctx.domain == "nursing"
        assert ctx.concern == ""  # default
        assert len(ctx.papers) == 1

    def test_review_context_immutable(self):
        """ReviewContext is frozen -- attributes cannot be reassigned."""
        ctx = ReviewContext(
            papers=(),
            committee_members=(),
            rubric={},
            domain="nursing",
        )
        with pytest.raises(AttributeError):
            ctx.domain = "other"  # type: ignore[misc]

    def test_review_context_concern_default(self):
        """concern defaults to empty string."""
        ctx = ReviewContext(papers=(), committee_members=(), rubric={}, domain="nursing")
        assert ctx.concern == ""

    def test_review_context_concern_provided(self):
        """concern can be explicitly set."""
        ctx = ReviewContext(
            papers=(), committee_members=(), rubric={}, domain="nursing", concern="methodology"
        )
        assert ctx.concern == "methodology"

    def test_to_dict_returns_dict(self):
        """to_dict returns a plain dict."""
        ctx = ReviewContext(
            papers=({"title": "P1", "abstract": "A1"},),
            committee_members=({"name": "a", "display_name": "A", "focus": "F"},),
            rubric={"a": ("d1",)},
            domain="nursing",
            concern="stats",
        )
        result = ctx.to_dict()
        assert isinstance(result, dict)
        assert result["domain"] == "nursing"
        assert result["concern"] == "stats"

    def test_to_dict_converts_tuples_to_lists(self):
        """to_dict converts tuple fields to lists for JSON serialization."""
        ctx = ReviewContext(
            papers=({"title": "P1", "abstract": "A1"},),
            committee_members=({"name": "a", "display_name": "A", "focus": "F"},),
            rubric={"a": ("d1", "d2")},
            domain="nursing",
        )
        result = ctx.to_dict()
        assert isinstance(result["papers"], list)
        assert isinstance(result["committee_members"], list)
        rubric = result["rubric"]
        assert isinstance(rubric, dict)
        assert isinstance(rubric["a"], list)

    def test_to_dict_empty_context(self):
        """to_dict works with empty collections."""
        ctx = ReviewContext(papers=(), committee_members=(), rubric={}, domain="nursing")
        result = ctx.to_dict()
        assert result["papers"] == []
        assert result["committee_members"] == []
        assert result["rubric"] == {}


# ---------------------------------------------------------------------------
# 2. prepare_review_context MCP tool
# ---------------------------------------------------------------------------


class TestPrepareReviewContext:
    """Tests for prepare_review_context MCP tool."""

    @pytest.mark.asyncio
    async def test_returns_dict(self, sample_agent_spec):
        """prepare_review_context returns a dict."""
        from unittest.mock import AsyncMock, patch

        from academic_helper.tools.committee import prepare_review_context

        mock_papers = [
            {
                "title": "Related Paper",
                "abstract": "Related abstract",
                "doi": "10.1/r",
                "authors": ["A"],
                "year": 2024,
                "source": "test",
                "url": None,
                "citation_count": 3,
            },
        ]

        with (
            patch(
                "academic_helper.tools.committee.load_all_agents",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.select_committee",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.search_papers",
                new_callable=AsyncMock,
                return_value=mock_papers,
            ),
        ):
            result = await prepare_review_context("Test Paper", "Test abstract")

        assert isinstance(result, dict)
        assert "papers" in result
        assert "committee_members" in result
        assert "rubric" in result
        assert "domain" in result

    @pytest.mark.asyncio
    async def test_default_domain_is_nursing(self, sample_agent_spec):
        """Default domain is nursing."""
        from unittest.mock import AsyncMock, patch

        from academic_helper.tools.committee import prepare_review_context

        with (
            patch(
                "academic_helper.tools.committee.load_all_agents",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.select_committee",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.search_papers",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await prepare_review_context("Test", "Abstract")

        assert result["domain"] == "nursing"

    @pytest.mark.asyncio
    async def test_concern_passed_through(self, sample_agent_spec):
        """User concern is included in the result."""
        from unittest.mock import AsyncMock, patch

        from academic_helper.tools.committee import prepare_review_context

        with (
            patch(
                "academic_helper.tools.committee.load_all_agents",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.select_committee",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.search_papers",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await prepare_review_context("Test", "Abstract", concern="\u7d71\u8a08\u65b9\u6cd5")

        assert result["concern"] == "\u7d71\u8a08\u65b9\u6cd5"

    @pytest.mark.asyncio
    async def test_papers_slim_format(self, sample_agent_spec):
        """Related papers are slimmed to title + abstract only."""
        from unittest.mock import AsyncMock, patch

        from academic_helper.tools.committee import prepare_review_context

        mock_papers = [
            {
                "title": "P1",
                "abstract": "A1",
                "doi": "10.1/x",
                "authors": ["X"],
                "year": 2024,
                "source": "s",
                "url": None,
                "citation_count": 10,
            },
        ]

        with (
            patch(
                "academic_helper.tools.committee.load_all_agents",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.select_committee",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.search_papers",
                new_callable=AsyncMock,
                return_value=mock_papers,
            ),
        ):
            result = await prepare_review_context("Test", "Abstract")

        paper = result["papers"][0]
        assert set(paper.keys()) == {"title", "abstract"}

    @pytest.mark.asyncio
    async def test_committee_member_format(self, sample_agent_spec):
        """Committee members have name, display_name, focus."""
        from unittest.mock import AsyncMock, patch

        from academic_helper.tools.committee import prepare_review_context

        with (
            patch(
                "academic_helper.tools.committee.load_all_agents",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.select_committee",
                return_value=(sample_agent_spec,),
            ),
            patch(
                "academic_helper.tools.committee.search_papers",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await prepare_review_context("Test", "Abstract")

        member = result["committee_members"][0]
        assert set(member.keys()) == {"name", "display_name", "focus"}

    def test_is_async(self):
        """prepare_review_context is an async function."""
        from academic_helper.tools.committee import prepare_review_context

        assert inspect.iscoroutinefunction(prepare_review_context)
