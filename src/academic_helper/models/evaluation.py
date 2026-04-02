"""Evaluation models — AgentSpec, AgentScore, EvaluationResult."""

from dataclasses import dataclass

from academic_helper.models.paper import Paper


@dataclass(frozen=True)
class AgentSpec:
    """Typed contract for a thesis committee agent definition."""

    name: str
    display_name: str
    focus: str
    scoring_dimensions: tuple[str, ...]
    prompt_template: str
    default_for: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentScore:
    """Score from a single agent evaluation."""

    agent_name: str
    score: float
    rationale: str


@dataclass(frozen=True)
class EvaluationResult:
    """Immutable result from the thesis defense committee evaluation."""

    paper: Paper
    scores: tuple[AgentScore, ...]
    overall_score: float
    recommendation: str
