"""Tests for data models — Paper, DomainProfile, AgentSpec, SourceResult."""

import pytest


class TestPaper:
    def test_paper_creation(self, sample_paper):
        assert sample_paper.title == "Effect of AI-Assisted Nursing on Elderly Care"
        assert sample_paper.doi == "10.1234/test.2024.001"
        assert sample_paper.year == 2024
        assert sample_paper.source == "openalex"

    def test_paper_authors_is_tuple(self, sample_paper):
        assert isinstance(sample_paper.authors, tuple)
        assert len(sample_paper.authors) == 2

    def test_paper_is_frozen(self, sample_paper):
        with pytest.raises(AttributeError):
            sample_paper.title = "modified"

    def test_paper_is_hashable(self, sample_paper):
        paper_set = {sample_paper}
        assert sample_paper in paper_set

    def test_paper_optional_fields(self):
        from academic_helper.models.paper import Paper

        p = Paper(
            title="Minimal Paper",
            abstract="",
            doi=None,
            authors=(),
            year=None,
            source="crossref",
            url=None,
            citation_count=0,
        )
        assert p.doi is None
        assert p.year is None
        assert p.url is None


class TestDomainProfile:
    def test_nursing_tw_preset(self, sample_domain):
        assert sample_domain.field == "nursing"
        assert sample_domain.locale == "zh-TW"
        assert sample_domain.citation_style == "APA7"

    def test_domain_profile_is_frozen(self, sample_domain):
        with pytest.raises(AttributeError):
            sample_domain.field = "education"

    def test_domain_profile_methodology_norms(self, sample_domain):
        assert isinstance(sample_domain.methodology_norms, tuple)
        assert "qualitative" in sample_domain.methodology_norms

    def test_domain_profile_keywords_hint(self, sample_domain):
        assert isinstance(sample_domain.keywords_hint, tuple)


class TestAgentSpec:
    def test_agent_spec_creation(self, sample_agent_spec):
        assert sample_agent_spec.name == "methodology-reviewer"
        assert sample_agent_spec.display_name == "方法論審查員"

    def test_agent_spec_is_frozen(self, sample_agent_spec):
        with pytest.raises(AttributeError):
            sample_agent_spec.name = "changed"

    def test_agent_spec_scoring_dimensions(self, sample_agent_spec):
        assert isinstance(sample_agent_spec.scoring_dimensions, tuple)
        assert len(sample_agent_spec.scoring_dimensions) == 4

    def test_agent_spec_default_for(self):
        from academic_helper.models.evaluation import AgentSpec

        spec = AgentSpec(
            name="ethics",
            display_name="倫理審查員",
            focus="Ethics review",
            scoring_dimensions=("consent", "privacy"),
            prompt_template="...",
            default_for=("nursing", "medicine"),
        )
        assert spec.default_for == ("nursing", "medicine")

    def test_agent_spec_default_for_empty(self, sample_agent_spec):
        assert sample_agent_spec.default_for == ()


class TestSourceResult:
    def test_source_result_success(self, sample_paper):
        from academic_helper.models.source import SourceResult

        result = SourceResult(
            source_name="openalex",
            papers=(sample_paper,),
        )
        assert result.error is None
        assert len(result.papers) == 1

    def test_source_result_failure(self):
        from academic_helper.models.source import SourceResult

        result = SourceResult(
            source_name="semantic_scholar",
            error="Rate limit exceeded",
        )
        assert result.error == "Rate limit exceeded"
        assert result.papers == ()

    def test_source_result_is_frozen(self, sample_paper):
        from academic_helper.models.source import SourceResult

        result = SourceResult(source_name="test", papers=(sample_paper,))
        with pytest.raises(AttributeError):
            result.source_name = "changed"

    def test_source_result_elapsed_ms(self, sample_paper):
        from academic_helper.models.source import SourceResult

        result = SourceResult(
            source_name="openalex",
            papers=(sample_paper,),
            elapsed_ms=150,
        )
        assert result.elapsed_ms == 150


class TestEvaluationResult:
    def test_evaluation_result(self, sample_paper):
        from academic_helper.models.evaluation import AgentScore, EvaluationResult

        score = AgentScore(agent_name="methodology", score=0.85, rationale="Good design")
        result = EvaluationResult(
            paper=sample_paper,
            scores=(score,),
            overall_score=0.85,
            recommendation="Accept with minor revisions",
        )
        assert result.overall_score == 0.85
        assert len(result.scores) == 1

    def test_evaluation_result_is_frozen(self, sample_paper):
        from academic_helper.models.evaluation import AgentScore, EvaluationResult

        score = AgentScore(agent_name="test", score=0.5, rationale="ok")
        result = EvaluationResult(
            paper=sample_paper,
            scores=(score,),
            overall_score=0.5,
            recommendation="Revise",
        )
        with pytest.raises(AttributeError):
            result.overall_score = 1.0
