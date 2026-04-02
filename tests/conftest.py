"""Shared test fixtures."""

import pytest

from academic_helper.models.paper import Paper
from academic_helper.models.domain import DomainProfile, NURSING_TW
from academic_helper.models.committee import ReviewContext
from academic_helper.models.evaluation import AgentSpec


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        title="Effect of AI-Assisted Nursing on Elderly Care",
        abstract="This study examines the impact of artificial intelligence...",
        doi="10.1234/test.2024.001",
        authors=("Wang, L.", "Chen, M."),
        year=2024,
        source="openalex",
        url="https://doi.org/10.1234/test.2024.001",
        citation_count=15,
    )


@pytest.fixture
def sample_domain() -> DomainProfile:
    return NURSING_TW


@pytest.fixture
def sample_agent_spec() -> AgentSpec:
    return AgentSpec(
        name="methodology-reviewer",
        display_name="方法論審查員",
        focus="Evaluates research design, sampling strategy, validity, and reliability.",
        scoring_dimensions=("research_design", "sampling", "validity", "reliability"),
        prompt_template="Evaluate the methodology of the following paper...",
    )


@pytest.fixture
def sample_review_context() -> ReviewContext:
    return ReviewContext(
        papers=({"title": "Related Paper", "abstract": "Related abstract"},),
        committee_members=(
            {
                "name": "methodology-reviewer",
                "display_name": "方法論審查員",
                "focus": "Research methodology",
            },
        ),
        rubric={"methodology-reviewer": ("research_design", "sampling", "validity")},
        domain="nursing",
        concern="統計分析方法",
    )
