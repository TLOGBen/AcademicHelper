"""Prompt template functions for paper content evaluation."""

from __future__ import annotations

from academic_helper.models.paper import Paper


def methodology_rubric(paper: Paper) -> str:
    """Generate a methodology evaluation prompt for the given paper."""
    return (
        "Methodology Evaluation Rubric\n"
        "==============================\n"
        f"Title: {paper.title}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        "Please evaluate the following methodology dimensions:\n"
        "1. Research design appropriateness\n"
        "2. Sampling strategy validity\n"
        "3. Internal and external validity\n"
        "4. Reliability and reproducibility\n"
        "5. Alignment of method to research question\n"
    )


def fact_vs_hypothesis(paper: Paper) -> str:
    """Generate a fact-vs-hypothesis distinction prompt for the given paper."""
    return (
        "Fact vs. Hypothesis Analysis\n"
        "=============================\n"
        f"Title: {paper.title}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        "Identify and classify the following in the paper:\n"
        "1. Established empirical facts cited by the authors\n"
        "2. Hypotheses or propositions to be tested\n"
        "3. Inferences drawn from evidence\n"
        "4. Speculation or unsupported claims\n"
        "5. Conclusions that exceed the data warrant\n"
    )


def logical_chain(paper: Paper) -> str:
    """Generate a logical-chain tracing prompt for the given paper."""
    return (
        "Logical Chain Analysis\n"
        "======================\n"
        f"Title: {paper.title}\n\n"
        f"Abstract:\n{paper.abstract}\n\n"
        "Trace the argumentative structure of the paper:\n"
        "1. Central thesis or research question\n"
        "2. Supporting premises and their evidence base\n"
        "3. Logical connections between premises and conclusions\n"
        "4. Potential gaps or non-sequiturs in the reasoning\n"
        "5. Overall logical coherence and strength of argument\n"
    )
