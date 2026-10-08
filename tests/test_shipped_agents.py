"""Regression checks for real agent assets and self-contained review context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from academic_helper.tools import committee, evaluate


SHIPPED_NAMES = {
    "cassandra",
    "data-quality-reviewer",
    "diogenes",
    "ethics-reviewer",
    "literature-reviewer",
    "methodology-reviewer",
    "practical-reviewer",
    "socrates",
}


def write_agent(path: Path, *, body: str = "Review the provided evidence.", **changes) -> Path:
    data = {
        "name": "test-reviewer",
        "display_name": "測試審查員",
        "focus": "Check the supplied research evidence.",
        "scoring_dimensions": ["evidence_quality"],
        "default_for": ["nursing"],
    }
    data.update(changes)
    path.write_text(f"---\n{yaml.safe_dump(data)}---\n\n{body}\n", encoding="utf-8")
    return path


def test_loads_markdown_body_as_prompt(tmp_path: Path):
    body = "# Review instructions\n\nAssess evidence and mark unavailable information."
    path = write_agent(tmp_path / "reviewer.md", body=body)
    assert evaluate.load_agent_spec(path).prompt_template == body


def test_explicit_prompt_template_takes_precedence(tmp_path: Path):
    path = write_agent(tmp_path / "reviewer.md", prompt_template="Review {title}.")
    assert evaluate.load_agent_spec(path).prompt_template == "Review {title}."


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", 1),
        ("display_name", " "),
        ("focus", None),
        ("scoring_dimensions", "evidence_quality"),
        ("scoring_dimensions", []),
        ("scoring_dimensions", ["evidence_quality", None]),
        ("default_for", "nursing"),
        ("prompt_template", " "),
        ("prompt_template", None),
    ],
)
def test_rejects_invalid_agent_metadata(tmp_path: Path, field: str, value):
    path = write_agent(tmp_path / "invalid.md", **{field: value})
    with pytest.raises(ValueError, match=field):
        evaluate.load_agent_spec(path)


def test_rejects_agent_without_a_prompt_or_body(tmp_path: Path):
    path = write_agent(tmp_path / "invalid.md", body="")
    with pytest.raises(ValueError, match="prompt_template"):
        evaluate.load_agent_spec(path)


def test_agent_path_resolution_handles_shallow_module_location(monkeypatch):
    monkeypatch.setattr(evaluate, "__file__", "/evaluate.py")
    assert evaluate._resolve_agents_dir() == Path("/agents")


def test_shipped_agents_are_complete_and_load_from_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agents = evaluate.load_all_agents(evaluate._agents_dir)
    assert {agent.name for agent in agents} == SHIPPED_NAMES
    for agent in agents:
        assert agent.focus.strip()
        assert agent.scoring_dimensions
        assert agent.prompt_template.strip()
        assert isinstance(agent.default_for, tuple)
        # The returned prompt is the authoritative shipped Markdown body.
        source = (evaluate._agents_dir / f"{agent.name}.md").read_text(encoding="utf-8")
        assert source.rstrip().endswith(agent.prompt_template)


def test_evaluate_paper_works_with_real_agents(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = evaluate.evaluate_paper("Target title", "Target abstract", domain="nursing")
    assert result["paper_title"] == "Target title"
    assert result["paper_abstract"] == "Target abstract"
    assert result["domain"] == "nursing"
    assert len(result["committee_members"]) == 5
    assert {member["name"] for member in result["committee_members"]} == {
        "data-quality-reviewer", "ethics-reviewer", "literature-reviewer",
        "methodology-reviewer", "practical-reviewer",
    }
    assert result["evaluation_dimensions"]
    for member in result["committee_members"]:
        assert member["prompt_template"].strip()
        assert member["scoring_dimensions"] == result["rubric"][member["name"]]
    json.dumps(result)


@pytest.mark.asyncio
async def test_committee_returns_shipped_review_prompts_and_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    queries = []

    async def search_stub(query, limit):
        queries.append((query, limit))
        return [
            {
                "title": "Related study", "abstract": "Supporting evidence",
                "doi": "10.1/x", "url": "https://doi.org/10.1/x",
                "year": 2025, "source": "test", "citation_count": 10,
            },
            {"title": "No identifiers", "abstract": "Metadata is unavailable"},
        ]

    # Only the remote search is stubbed; agent files and selection are real.
    monkeypatch.setattr(committee, "search_papers", search_stub)
    result = await committee.prepare_review_context(
        "Target title", "Target abstract", concern="Sampling"
    )
    assert queries == [("Target title Target abstract", 5)]
    assert result["paper_title"] == "Target title"
    assert result["paper_abstract"] == "Target abstract"
    assert result["concern"] == "Sampling"
    assert result["papers"] == [
        {
            "title": "Related study", "abstract": "Supporting evidence",
            "doi": "10.1/x", "url": "https://doi.org/10.1/x",
            "year": 2025, "source": "test",
        },
        {"title": "No identifiers", "abstract": "Metadata is unavailable"},
    ]
    assert len(result["committee_members"]) == 5
    by_name = {agent.name: agent for agent in evaluate.load_all_agents(evaluate._agents_dir)}
    for member in result["committee_members"]:
        assert member["prompt_template"] == by_name[member["name"]].prompt_template
        assert member["scoring_dimensions"] == result["rubric"][member["name"]]
        assert member["scoring_dimensions"]
    json.dumps(result)
