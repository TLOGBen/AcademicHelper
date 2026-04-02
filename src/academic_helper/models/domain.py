"""DomainProfile — abstraction for academic discipline characteristics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    """Immutable profile describing an academic discipline's conventions."""

    field: str
    locale: str
    methodology_norms: tuple[str, ...]
    citation_style: str
    keywords_hint: tuple[str, ...] = ()


NURSING_TW = DomainProfile(
    field="nursing",
    locale="zh-TW",
    methodology_norms=("quantitative", "qualitative", "mixed-methods"),
    citation_style="APA7",
    keywords_hint=("護理", "elderly care", "clinical nursing", "長照"),
)
