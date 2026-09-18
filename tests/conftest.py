"""Shared fixtures.

The suite never calls OpenAI: it runs offline, for free, and gives the same answer
every time. A fake key is set before the app is imported, since config fails loudly
without one.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


@pytest.fixture
def chunk() -> Document:
    """A chunk as it looks after ingest: text plus a locator to cite."""
    return Document(
        page_content="Production infrastructure is hosted on GCP. Backups run nightly.",
        metadata={"source": "report.pdf", "page": 17, "locator": "page 17"},
    )


@pytest.fixture
def hits(chunk: Document) -> list[tuple[Document, float]]:
    return [(chunk, 0.42)]


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    """A three-page PDF with a repeated header, built in memory by pypdf.

    Small on purpose: the 84-page sample belongs in manual runs, not in a test suite.
    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)

    path = tmp_path / "tiny.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    return path


@pytest.fixture
def records_json(tmp_path: Path) -> Path:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a1", "question": "Where are your data centres?", "answer": "US Central, on GCP."},
                {"id": "a2", "question": "Do you encrypt backups?", "answer": "Yes, AES-256.", "tags": ["crypto", "backup"]},
            ]
        ),
        encoding="utf-8",
    )
    return path
