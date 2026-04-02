"""PDF extraction tool — extract title, abstract, and authors from a PDF file."""

from __future__ import annotations

import re


def _read_pdf_text(path: str) -> str:
    """Read and return the full text content of a PDF file.

    This is the sole I/O seam for PDF reading; mock this in tests.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        import pypdf  # type: ignore[import]

        reader = pypdf.PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        with open(path, "rb") as f:
            raw = f.read()
        return raw.decode("utf-8", errors="replace")


def _extract_title(text: str) -> str:
    """Extract title from PDF text."""
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^Title:\s*(.+)", stripped, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Fallback: first non-empty line
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_abstract(text: str) -> str:
    """Extract abstract from PDF text."""
    match = re.search(
        r"Abstract[:\s]+(.+?)(?:\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def _extract_authors(text: str) -> list[str]:
    """Extract authors from PDF text."""
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"^Authors?:\s*(.+)", stripped, re.IGNORECASE)
        if match:
            raw = match.group(1)
            parts = [a.strip() for a in re.split(r"[;,]", raw) if a.strip()]
            return parts
    return []


async def extract_paper_from_pdf(pdf_path: str) -> dict:
    """Extract title, abstract, and authors from a PDF file.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        Dict with keys: title, abstract, authors.

    Raises:
        FileNotFoundError: Propagated from _read_pdf_text when file is missing.
    """
    text = _read_pdf_text(pdf_path)
    return {
        "title": _extract_title(text),
        "abstract": _extract_abstract(text),
        "authors": _extract_authors(text),
    }
