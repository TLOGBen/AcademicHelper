"""Committee review tools — prepare_review_context for thesis committee evaluation."""

from __future__ import annotations

from academic_helper.models.committee import ReviewContext
from academic_helper.tools.evaluate import (
    AGENTS_DIR,
    agent_review_metadata,
    domain_profile_for,
    load_all_agents,
    select_committee,
)
from academic_helper.tools.search import search_papers


async def prepare_review_context(
    paper_title: str,
    paper_abstract: str,
    domain: str = "nursing",
    concern: str = "",
) -> dict:
    """Prepare a pre-fetched context bundle for thesis committee evaluation.

    Searches for related literature, selects committee members, and returns
    a context package that can be used to dispatch committee member agents.
    """
    agents = load_all_agents(AGENTS_DIR)
    profile = domain_profile_for(domain)
    committee = select_committee(profile, agents, max_size=5)

    query = f"{paper_title} {paper_abstract[:200]}"
    raw_papers = await search_papers(query, limit=5)
    papers_slim = [
        {
            "title": p["title"],
            "abstract": p["abstract"],
            **{key: p[key] for key in ("doi", "url", "year", "source") if key in p},
        }
        for p in raw_papers
    ]

    members = [agent_review_metadata(a) for a in committee]

    rubric = {a.name: a.scoring_dimensions for a in committee}

    context = ReviewContext(
        papers=tuple(papers_slim),
        committee_members=tuple(members),
        rubric=rubric,
        domain=domain,
        concern=concern,
    )
    result = context.to_dict()
    result["paper_title"] = paper_title
    result["paper_abstract"] = paper_abstract
    return result


def register(mcp) -> None:
    """Register committee review tools onto the given FastMCP instance."""
    mcp.tool()(prepare_review_context)
