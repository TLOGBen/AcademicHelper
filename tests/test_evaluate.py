"""Tests for L006: Content Evaluation — agent loading, committee selection, prompts.

All tests are written in TDD RED phase: the implementation does not yet exist,
so these tests are expected to FAIL when first run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from academic_helper.models.domain import NURSING_TW, DomainProfile
from academic_helper.models.evaluation import AgentSpec
from academic_helper.models.paper import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_FRONTMATTER = """\
---
name: methodology-reviewer
display_name: 方法論審查員
focus: Evaluates research design, sampling strategy, validity, and reliability.
scoring_dimensions:
  - research_design
  - sampling
  - validity
  - reliability
default_for:
  - nursing
  - medicine
prompt_template: |
  Evaluate the methodology of the following paper:
  Title: {title}
  Abstract: {abstract}
---

# Methodology Reviewer

This agent evaluates the research design and methodology.
"""

VALID_FRONTMATTER_NO_DEFAULT = """\
---
name: stats-reviewer
display_name: 統計審查員
focus: Evaluates statistical methods and data analysis.
scoring_dimensions:
  - statistical_method
  - sample_size
  - effect_size
default_for: []
prompt_template: |
  Evaluate the statistics of: {title}
---
"""

INVALID_NO_FRONTMATTER = """\
# Just a plain markdown file

No YAML frontmatter here at all.
"""

INVALID_MISSING_REQUIRED_FIELD = """\
---
name: incomplete-agent
display_name: 不完整審查員
focus: Missing required fields below
---
"""

INVALID_YAML_SYNTAX = """\
---
name: bad-yaml
display_name: [unclosed bracket
focus: broken yaml here
---
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_agent_md(tmp_path: Path) -> Path:
    """A single valid agent .md file."""
    p = tmp_path / "methodology-reviewer.md"
    p.write_text(VALID_FRONTMATTER, encoding="utf-8")
    return p


@pytest.fixture
def agent_with_no_default(tmp_path: Path) -> Path:
    """A valid agent .md file with empty default_for."""
    p = tmp_path / "stats-reviewer.md"
    p.write_text(VALID_FRONTMATTER_NO_DEFAULT, encoding="utf-8")
    return p


@pytest.fixture
def agents_dir(tmp_path: Path) -> Path:
    """A directory with several agent .md files."""
    agents = tmp_path / "agents"
    agents.mkdir()

    # nursing + medicine specialist
    (agents / "methodology-reviewer.md").write_text(
        """\
---
name: methodology-reviewer
display_name: 方法論審查員
focus: Evaluates research design and methodology.
scoring_dimensions:
  - research_design
  - sampling
  - validity
default_for:
  - nursing
  - medicine
prompt_template: |
  Methodology check for {title}
---
""",
        encoding="utf-8",
    )

    # nursing specialist only
    (agents / "ethics-reviewer.md").write_text(
        """\
---
name: ethics-reviewer
display_name: 倫理審查員
focus: Evaluates ethical considerations in nursing research.
scoring_dimensions:
  - informed_consent
  - privacy
  - risk_management
default_for:
  - nursing
prompt_template: |
  Ethics review for {title}
---
""",
        encoding="utf-8",
    )

    # education specialist — not in nursing
    (agents / "curriculum-reviewer.md").write_text(
        """\
---
name: curriculum-reviewer
display_name: 課程審查員
focus: Evaluates curriculum design and pedagogical approaches.
scoring_dimensions:
  - curriculum_design
  - learning_outcomes
default_for:
  - education
prompt_template: |
  Curriculum review for {title}
---
""",
        encoding="utf-8",
    )

    # generic — no default_for
    (agents / "stats-reviewer.md").write_text(
        """\
---
name: stats-reviewer
display_name: 統計審查員
focus: Evaluates statistical methods and data analysis.
scoring_dimensions:
  - statistical_method
  - sample_size
default_for: []
prompt_template: |
  Statistics review for {title}
---
""",
        encoding="utf-8",
    )

    # another generic
    (agents / "writing-reviewer.md").write_text(
        """\
---
name: writing-reviewer
display_name: 寫作審查員
focus: Evaluates writing clarity and academic style.
scoring_dimensions:
  - clarity
  - academic_style
default_for: []
prompt_template: |
  Writing review for {title}
---
""",
        encoding="utf-8",
    )

    # non-.md file — must be ignored
    (agents / "README.txt").write_text("ignore me", encoding="utf-8")

    return agents


@pytest.fixture
def sample_paper() -> Paper:
    return Paper(
        title="Effect of AI-Assisted Nursing on Elderly Care",
        abstract="This study examines the impact of AI in clinical nursing settings.",
        doi="10.1234/test.2024.001",
        authors=("Wang, L.", "Chen, M."),
        year=2024,
        source="openalex",
        url="https://doi.org/10.1234/test.2024.001",
        citation_count=15,
    )


# ---------------------------------------------------------------------------
# 1. Agent Loader — load_agent_spec
# ---------------------------------------------------------------------------


class TestLoadAgentSpec:
    def test_returns_agent_spec_instance(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert isinstance(result, AgentSpec)

    def test_parses_name(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert result.name == "methodology-reviewer"

    def test_parses_display_name_in_chinese(self, valid_agent_md: Path) -> None:
        """display_name must use user-goal language (Chinese for zh-TW locale)."""
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert result.display_name == "方法論審查員"

    def test_parses_focus(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert "research design" in result.focus

    def test_parses_scoring_dimensions_as_tuple(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert isinstance(result.scoring_dimensions, tuple)
        assert "research_design" in result.scoring_dimensions
        assert "sampling" in result.scoring_dimensions

    def test_parses_default_for_as_tuple(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert isinstance(result.default_for, tuple)
        assert "nursing" in result.default_for
        assert "medicine" in result.default_for

    def test_empty_default_for_yields_empty_tuple(self, agent_with_no_default: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(agent_with_no_default)
        assert result.default_for == ()

    def test_parses_prompt_template(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert len(result.prompt_template) > 0

    def test_result_is_frozen(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        with pytest.raises(AttributeError):
            result.name = "changed"  # type: ignore[misc]

    def test_raises_value_error_for_missing_frontmatter(self, tmp_path: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        bad_file = tmp_path / "bad.md"
        bad_file.write_text(INVALID_NO_FRONTMATTER, encoding="utf-8")
        with pytest.raises(ValueError, match="frontmatter"):
            load_agent_spec(bad_file)

    def test_raises_value_error_for_missing_required_field(self, tmp_path: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        bad_file = tmp_path / "incomplete.md"
        bad_file.write_text(INVALID_MISSING_REQUIRED_FIELD, encoding="utf-8")
        with pytest.raises(ValueError):
            load_agent_spec(bad_file)

    def test_raises_value_error_for_invalid_yaml(self, tmp_path: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        bad_file = tmp_path / "invalid.md"
        bad_file.write_text(INVALID_YAML_SYNTAX, encoding="utf-8")
        with pytest.raises(ValueError, match="YAML"):
            load_agent_spec(bad_file)

    def test_raises_file_not_found_for_missing_file(self, tmp_path: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        nonexistent = tmp_path / "ghost.md"
        with pytest.raises(FileNotFoundError):
            load_agent_spec(nonexistent)


# ---------------------------------------------------------------------------
# 2. Agent Loader — load_all_agents
# ---------------------------------------------------------------------------


class TestLoadAllAgents:
    def test_returns_tuple(self, agents_dir: Path) -> None:
        from academic_helper.tools.evaluate import load_all_agents

        result = load_all_agents(agents_dir)
        assert isinstance(result, tuple)

    def test_loads_only_md_files(self, agents_dir: Path) -> None:
        """The README.txt file must not be loaded."""
        from academic_helper.tools.evaluate import load_all_agents

        result = load_all_agents(agents_dir)
        names = {a.name for a in result}
        assert "README" not in names

    def test_loads_correct_count(self, agents_dir: Path) -> None:
        """5 .md files are present; 1 non-.md file must be skipped."""
        from academic_helper.tools.evaluate import load_all_agents

        result = load_all_agents(agents_dir)
        assert len(result) == 5

    def test_all_elements_are_agent_spec(self, agents_dir: Path) -> None:
        from academic_helper.tools.evaluate import load_all_agents

        result = load_all_agents(agents_dir)
        for agent in result:
            assert isinstance(agent, AgentSpec)

    def test_result_is_frozen_tuple(self, agents_dir: Path) -> None:
        """Result must be a tuple (immutable collection)."""
        from academic_helper.tools.evaluate import load_all_agents

        result = load_all_agents(agents_dir)
        assert isinstance(result, tuple)
        with pytest.raises(AttributeError):
            result.append(None)  # type: ignore[attr-defined]

    def test_raises_for_missing_directory(self, tmp_path: Path) -> None:
        from academic_helper.tools.evaluate import load_all_agents

        missing = tmp_path / "no_such_dir"
        with pytest.raises((FileNotFoundError, NotADirectoryError)):
            load_all_agents(missing)

    def test_empty_directory_returns_empty_tuple(self, tmp_path: Path) -> None:
        from academic_helper.tools.evaluate import load_all_agents

        empty = tmp_path / "empty_agents"
        empty.mkdir()
        result = load_all_agents(empty)
        assert result == ()


# ---------------------------------------------------------------------------
# 3. Committee Selection — select_committee
# ---------------------------------------------------------------------------


@pytest.fixture
def all_agents(agents_dir: Path) -> tuple[AgentSpec, ...]:
    from academic_helper.tools.evaluate import load_all_agents

    return load_all_agents(agents_dir)


class TestSelectCommittee:
    def test_returns_tuple(self, all_agents: tuple[AgentSpec, ...]) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents)
        assert isinstance(result, tuple)

    def test_prioritises_domain_matching_agents(
        self, all_agents: tuple[AgentSpec, ...]
    ) -> None:
        """Agents with default_for overlapping domain.field appear in result."""
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents)
        names = {a.name for a in result}
        # methodology-reviewer and ethics-reviewer both list "nursing" in default_for
        assert "methodology-reviewer" in names
        assert "ethics-reviewer" in names

    def test_does_not_include_non_domain_first_when_enough_specialists(
        self, all_agents: tuple[AgentSpec, ...]
    ) -> None:
        """curriculum-reviewer (education only) should not crowd out specialists."""
        from academic_helper.tools.evaluate import select_committee

        # With max_size=2, only domain specialists should fill slots
        result = select_committee(NURSING_TW, all_agents, max_size=2)
        names = {a.name for a in result}
        assert "curriculum-reviewer" not in names

    def test_fills_remaining_slots_from_non_domain_agents(
        self, all_agents: tuple[AgentSpec, ...]
    ) -> None:
        """When fewer specialists than max_size, fill remaining from others."""
        from academic_helper.tools.evaluate import select_committee

        # NURSING_TW has 2 specialists; with max_size=4, 2 fillers must be added
        result = select_committee(NURSING_TW, all_agents, max_size=4)
        assert len(result) == 4

    def test_does_not_exceed_max_size(self, all_agents: tuple[AgentSpec, ...]) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents, max_size=3)
        assert len(result) <= 3

    def test_result_sorted_by_name_for_determinism(
        self, all_agents: tuple[AgentSpec, ...]
    ) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents)
        names = [a.name for a in result]
        assert names == sorted(names)

    def test_result_is_tuple(self, all_agents: tuple[AgentSpec, ...]) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents)
        assert isinstance(result, tuple)

    def test_default_max_size_is_five(self, all_agents: tuple[AgentSpec, ...]) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents)
        assert len(result) <= 5

    def test_domain_with_no_matching_specialists(
        self, all_agents: tuple[AgentSpec, ...]
    ) -> None:
        """Domain with no specialist matches falls back entirely to fillers."""
        from academic_helper.tools.evaluate import select_committee

        obscure_domain = DomainProfile(
            field="veterinary",
            locale="en",
            methodology_norms=(),
            citation_style="APA7",
        )
        result = select_committee(obscure_domain, all_agents, max_size=2)
        assert len(result) == 2
        # No agent has default_for containing "veterinary"
        for agent in result:
            assert "veterinary" not in agent.default_for

    def test_empty_agents_returns_empty_tuple(self) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, ())
        assert result == ()

    def test_no_duplicate_agents_in_committee(
        self, all_agents: tuple[AgentSpec, ...]
    ) -> None:
        from academic_helper.tools.evaluate import select_committee

        result = select_committee(NURSING_TW, all_agents, max_size=5)
        names = [a.name for a in result]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# 4. Prompt Templates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_methodology_rubric_returns_string(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import methodology_rubric

        result = methodology_rubric(sample_paper)
        assert isinstance(result, str)

    def test_methodology_rubric_includes_title(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import methodology_rubric

        result = methodology_rubric(sample_paper)
        assert sample_paper.title in result

    def test_methodology_rubric_includes_abstract(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import methodology_rubric

        result = methodology_rubric(sample_paper)
        assert sample_paper.abstract in result

    def test_methodology_rubric_is_non_empty(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import methodology_rubric

        result = methodology_rubric(sample_paper)
        assert len(result.strip()) > 0

    def test_fact_vs_hypothesis_returns_string(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import fact_vs_hypothesis

        result = fact_vs_hypothesis(sample_paper)
        assert isinstance(result, str)

    def test_fact_vs_hypothesis_includes_title(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import fact_vs_hypothesis

        result = fact_vs_hypothesis(sample_paper)
        assert sample_paper.title in result

    def test_fact_vs_hypothesis_includes_abstract(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import fact_vs_hypothesis

        result = fact_vs_hypothesis(sample_paper)
        assert sample_paper.abstract in result

    def test_fact_vs_hypothesis_is_non_empty(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import fact_vs_hypothesis

        result = fact_vs_hypothesis(sample_paper)
        assert len(result.strip()) > 0

    def test_logical_chain_returns_string(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import logical_chain

        result = logical_chain(sample_paper)
        assert isinstance(result, str)

    def test_logical_chain_includes_title(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import logical_chain

        result = logical_chain(sample_paper)
        assert sample_paper.title in result

    def test_logical_chain_includes_abstract(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import logical_chain

        result = logical_chain(sample_paper)
        assert sample_paper.abstract in result

    def test_logical_chain_is_non_empty(self, sample_paper: Paper) -> None:
        from academic_helper.prompts.templates import logical_chain

        result = logical_chain(sample_paper)
        assert len(result.strip()) > 0

    def test_templates_produce_different_prompts(self, sample_paper: Paper) -> None:
        """Each template must be distinct — they serve different evaluation goals."""
        from academic_helper.prompts.templates import (
            fact_vs_hypothesis,
            logical_chain,
            methodology_rubric,
        )

        rubric = methodology_rubric(sample_paper)
        fvh = fact_vs_hypothesis(sample_paper)
        lc = logical_chain(sample_paper)
        assert rubric != fvh
        assert fvh != lc
        assert rubric != lc

    def test_methodology_rubric_does_not_mutate_paper(self, sample_paper: Paper) -> None:
        """Paper is frozen; template functions must not attempt mutation."""
        from academic_helper.prompts.templates import methodology_rubric

        original_title = sample_paper.title
        methodology_rubric(sample_paper)
        assert sample_paper.title == original_title


# ---------------------------------------------------------------------------
# 5. Agent .md file structure validation
# ---------------------------------------------------------------------------


class TestAgentMdFrontmatter:
    """Validate required frontmatter fields are present and correctly typed."""

    @pytest.mark.parametrize(
        "required_field",
        ["name", "display_name", "focus", "scoring_dimensions", "prompt_template"],
    )
    def test_required_field_present_in_valid_file(
        self, valid_agent_md: Path, required_field: str
    ) -> None:
        """After loading, the required field is populated."""
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert getattr(result, required_field, None) is not None

    def test_display_name_is_user_language(self, valid_agent_md: Path) -> None:
        """display_name must be user-goal language (Chinese), not system jargon."""
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        # Must contain at least one CJK character (Chinese)
        has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in result.display_name)
        assert has_chinese, f"display_name '{result.display_name}' has no Chinese characters"

    def test_scoring_dimensions_must_be_non_empty(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert len(result.scoring_dimensions) > 0

    def test_name_is_kebab_case(self, valid_agent_md: Path) -> None:
        """Agent names should use kebab-case slugs."""
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert " " not in result.name
        assert result.name == result.name.lower()

    def test_focus_is_non_empty_string(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert isinstance(result.focus, str)
        assert len(result.focus.strip()) > 0

    def test_prompt_template_is_non_empty_string(self, valid_agent_md: Path) -> None:
        from academic_helper.tools.evaluate import load_agent_spec

        result = load_agent_spec(valid_agent_md)
        assert isinstance(result.prompt_template, str)
        assert len(result.prompt_template.strip()) > 0


# ---------------------------------------------------------------------------
# 6. evaluate_paper tool (MCP registration + return shape)
# ---------------------------------------------------------------------------


class TestEvaluatePaperTool:
    """Tests for the evaluate_paper MCP tool function return structure.

    LLM calls are NOT made — we only verify the dict shape and committee
    selection logic the function wraps.
    """

    def test_evaluate_paper_returns_dict(self, agents_dir: Path) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract for testing purposes.",
                domain="nursing",
            )
        assert isinstance(result, dict)

    def test_evaluate_paper_contains_committee_members(self, agents_dir: Path) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
                domain="nursing",
            )
        assert "committee_members" in result

    def test_evaluate_paper_committee_members_is_list(self, agents_dir: Path) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
                domain="nursing",
            )
        assert isinstance(result["committee_members"], list)

    def test_evaluate_paper_contains_evaluation_dimensions(self, agents_dir: Path) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
                domain="nursing",
            )
        assert "evaluation_dimensions" in result

    def test_evaluate_paper_contains_rubric(self, agents_dir: Path) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
                domain="nursing",
            )
        assert "rubric" in result

    def test_evaluate_paper_committee_members_have_display_name(
        self, agents_dir: Path
    ) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
                domain="nursing",
            )
        for member in result["committee_members"]:
            assert "display_name" in member

    def test_evaluate_paper_default_domain_is_nursing(self, agents_dir: Path) -> None:
        from unittest.mock import patch

        from academic_helper.tools.evaluate import evaluate_paper

        with patch(
            "academic_helper.tools.evaluate._agents_dir",
            new=agents_dir,
        ):
            result_default = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
            )
            result_explicit = evaluate_paper(
                paper_title="Test Paper",
                paper_abstract="An abstract.",
                domain="nursing",
            )
        # Default should produce the same committee as explicit nursing
        assert result_default["committee_members"] == result_explicit["committee_members"]

    def test_register_adds_evaluate_paper_tool(self) -> None:
        """register(mcp) must register a tool named 'evaluate_paper'."""
        from unittest.mock import MagicMock

        from academic_helper.tools.evaluate import register

        mock_mcp = MagicMock()
        register(mock_mcp)

        # mcp.tool() should have been called (used as decorator)
        assert mock_mcp.tool.called
