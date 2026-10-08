"""Evaluation tools — agent loading, committee selection, MCP tool registration."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from academic_helper.models.domain import NURSING_TW, DomainProfile
from academic_helper.models.evaluation import AgentSpec


def _resolve_agents_dir() -> Path:
    """Find bundled agent definitions, or their authoritative source in a checkout."""
    module_path = Path(__file__).resolve()
    bundled_dir = module_path.parent.parent / "agents"
    if bundled_dir.is_dir():
        return bundled_dir

    # An editable installation imports this module directly from src/. Resolve
    # against its location rather than requiring the caller to run in the repo.
    if len(module_path.parents) > 3 and module_path.parents[2].name == "src":
        checkout_dir = module_path.parents[3] / "agents"
        if checkout_dir.is_dir():
            return checkout_dir
    return bundled_dir


# Retain both patch seams for callers and tests of the two context tools.
_agents_dir: Path = _resolve_agents_dir()
AGENTS_DIR: Path = _agents_dir

_REQUIRED_FIELDS = ("name", "display_name", "focus", "scoring_dimensions", "prompt_template")


def _parse_frontmatter(path: Path) -> dict:
    """Parse YAML frontmatter from a markdown file.

    Raises:
        FileNotFoundError: file does not exist.
        ValueError: frontmatter is missing, malformed, or has bad YAML.
    """
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*(\n|$)", text, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {path}")
    raw = match.group(1)
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"YAML frontmatter in {path} must be a mapping")
    # Agent Markdown is the authoritative prompt. Older definitions and external
    # configurations may keep an explicit prompt_template in the frontmatter.
    if "prompt_template" not in data:
        data["prompt_template"] = text[match.end() :].strip()
    return data


def load_agent_spec(path: Path) -> AgentSpec:
    """Load a single agent spec from a markdown file with YAML frontmatter.

    Raises:
        FileNotFoundError: file does not exist.
        ValueError: frontmatter is missing, required field is absent, or YAML is bad.
    """
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")

    data = _parse_frontmatter(path)

    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing required fields in {path}: {missing}")

    for field in ("name", "display_name", "focus", "prompt_template"):
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Field {field} in {path} must be a non-empty string")

    raw_dims = data["scoring_dimensions"]
    if (
        not isinstance(raw_dims, list)
        or not raw_dims
        or any(not isinstance(dim, str) or not dim.strip() for dim in raw_dims)
    ):
        raise ValueError(f"Field scoring_dimensions in {path} must be a non-empty list of strings")
    scoring_dimensions = tuple(raw_dims)

    raw_default = data.get("default_for")
    if raw_default is None:
        raw_default = []
    if not isinstance(raw_default, list) or any(
        not isinstance(field, str) or not field.strip() for field in raw_default
    ):
        raise ValueError(f"Field default_for in {path} must be a list of strings")
    default_for = tuple(raw_default)

    return AgentSpec(
        name=data["name"],
        display_name=data["display_name"],
        focus=data["focus"],
        scoring_dimensions=scoring_dimensions,
        prompt_template=data["prompt_template"],
        default_for=default_for,
    )


def load_all_agents(agents_dir: Path) -> tuple[AgentSpec, ...]:
    """Load all agent specs from .md files in a directory.

    Raises:
        FileNotFoundError: directory does not exist.
        NotADirectoryError: path is not a directory.
    """
    if not agents_dir.exists():
        raise FileNotFoundError(f"Agents directory not found: {agents_dir}")
    if not agents_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {agents_dir}")

    md_files = sorted(agents_dir.glob("*.md"))
    return tuple(load_agent_spec(f) for f in md_files)


def select_committee(
    domain: DomainProfile,
    agents: tuple[AgentSpec, ...],
    max_size: int = 5,
) -> tuple[AgentSpec, ...]:
    """Select a review committee from available agents for the given domain.

    Priority: agents whose default_for contains domain.field.
    Backfill: remaining agents fill remaining slots up to max_size.
    Result is sorted by name for determinism and has no duplicates.
    """
    specialists = [a for a in agents if domain.field in a.default_for]
    others = [a for a in agents if domain.field not in a.default_for]

    selected = specialists[:max_size]
    remaining_slots = max_size - len(selected)
    if remaining_slots > 0:
        selected = selected + others[:remaining_slots]

    return tuple(sorted(selected, key=lambda a: a.name))


def domain_profile_for(domain: str) -> DomainProfile:
    """Return a DomainProfile for the given field name string."""
    if domain == NURSING_TW.field:
        return NURSING_TW
    return DomainProfile(
        field=domain,
        locale="en",
        methodology_norms=(),
        citation_style="APA7",
    )


def agent_review_metadata(agent: AgentSpec) -> dict:
    """Return enough of an agent definition to dispatch it without a local file."""
    return {
        "name": agent.name,
        "display_name": agent.display_name,
        "focus": agent.focus,
        "scoring_dimensions": list(agent.scoring_dimensions),
        "prompt_template": agent.prompt_template,
    }


def evaluate_paper(
    paper_title: str,
    paper_abstract: str,
    domain: str = "nursing",
) -> dict:
    """Build evaluation context for a paper (committee + dimensions + rubric).

    Does NOT call an LLM — returns structured metadata for downstream use.
    """
    agents = load_all_agents(_agents_dir)
    profile = domain_profile_for(domain)
    committee = select_committee(profile, agents)

    committee_members = [agent_review_metadata(a) for a in committee]
    evaluation_dimensions: list[str] = []
    for agent in committee:
        for dim in agent.scoring_dimensions:
            if dim not in evaluation_dimensions:
                evaluation_dimensions.append(dim)

    rubric = {
        a.name: list(a.scoring_dimensions)
        for a in committee
    }

    return {
        "paper_title": paper_title,
        "paper_abstract": paper_abstract,
        "domain": domain,
        "committee_members": committee_members,
        "evaluation_dimensions": evaluation_dimensions,
        "rubric": rubric,
    }


def register(mcp) -> None:
    """Register evaluate_paper as an MCP tool."""

    @mcp.tool()
    def evaluate_paper_tool(
        paper_title: str,
        paper_abstract: str,
        domain: str = "nursing",
    ) -> dict:
        """Prepare reviewer prompts and scoring dimensions for a paper; does not score it."""
        return evaluate_paper(paper_title, paper_abstract, domain)
