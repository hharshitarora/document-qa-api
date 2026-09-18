"""Loading, cleaning and splitting. No network, no model."""

import json
from pathlib import Path

import pytest

from app.errors import InputError
from app.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _find_running_header,
    _flatten,
    _normalise_whitespace,
    load,
    load_bytes,
    split,
)


class TestWhitespace:
    def test_collapses_layout_padding(self):
        # pypdf preserves visual layout, which cost 23% of the sample report's characters.
        assert _normalise_whitespace("A  Type  2   Report") == "A Type 2 Report"

    def test_collapses_newlines_and_tabs(self):
        assert _normalise_whitespace("line one\n\n\tline two") == "line one line two"

    def test_empty_stays_empty(self):
        assert _normalise_whitespace("   \n  ") == ""


class TestRunningHeader:
    def test_finds_a_header_shared_by_most_pages(self):
        header = "Confidential Annual Security Report 2026"
        pages = [f"{header} {number} distinct opening words for this page here" for number in range(10)]

        # The detector strips the longest shared prefix, so it may reach a word or two
        # past the header when pages begin similarly. It must never stop short.
        assert _find_running_header(pages).startswith(header)

    def test_cuts_at_a_word_boundary(self):
        """The scan steps in fixed jumps and once left every page starting with "ity"."""
        header = "A Type 2 Independent Service Auditor Report on Controls Relevant To Security"
        pages = [f"{header} {number} " + f"body{number} " * 40 for number in range(8)]

        found = _find_running_header(pages)

        assert found.startswith(header), "the whole header must be found, not part of it"
        assert not found.endswith(("Secur", "Securit")), "must not cut mid-word"
        # Every token kept is a complete word from the pages.
        assert all(token in pages[0].split() for token in found.split())

    def test_no_header_when_pages_differ(self):
        pages = [f"page {number} unique opening line and more text" * 5 for number in range(6)]
        assert _find_running_header(pages) == ""

    def test_too_few_pages_to_judge(self):
        assert _find_running_header(["same start here", "same start here"]) == ""


class TestSplit:
    def test_carries_metadata_onto_every_chunk(self):
        from langchain_core.documents import Document

        long_page = Document(
            page_content="sentence. " * 400,
            metadata={"source": "report.pdf", "page": 9, "locator": "page 9"},
        )

        chunks = split([long_page])

        assert len(chunks) > 1, "a 4000 character page should split"
        assert all(chunk.metadata["locator"] == "page 9" for chunk in chunks)
        assert all(len(chunk.page_content) <= CHUNK_SIZE for chunk in chunks)

    def test_chunks_overlap(self):
        from langchain_core.documents import Document

        body = " ".join(f"word{number}" for number in range(600))
        chunks = split([Document(page_content=body, metadata={"locator": "page 1"})])

        first_tail = chunks[0].page_content[-CHUNK_OVERLAP:]
        # Some of the first chunk's tail must reappear at the head of the second.
        assert any(word in chunks[1].page_content for word in first_tail.split()[:3])

    def test_short_page_stays_whole(self):
        from langchain_core.documents import Document

        page = Document(page_content="short body", metadata={"locator": "record 1"})
        assert len(split([page])) == 1


class TestFlatten:
    def test_object_becomes_key_value_lines(self):
        text = _flatten({"question": "Where?", "answer": "US Central"})
        assert "question: Where?" in text
        assert "answer: US Central" in text

    def test_nested_values_are_kept_as_json(self):
        text = _flatten({"tags": ["crypto", "backup"], "meta": {"confidence": "high"}})
        assert "crypto" in text and "backup" in text
        assert "confidence" in text

    def test_empty_fields_are_skipped(self):
        assert "blank" not in _flatten({"blank": "", "real": "value"})

    def test_scalars_and_none(self):
        assert _flatten("plain string") == "plain string"
        assert _flatten(None) == ""


class TestLoadJson:
    def test_one_document_per_record(self, records_json: Path):
        documents = load(records_json)

        assert len(documents) == 2
        assert [document.metadata["locator"] for document in documents] == ["record 1", "record 2"]
        assert "US Central" in documents[0].page_content

    def test_single_object_becomes_one_document(self, tmp_path: Path):
        path = tmp_path / "one.json"
        path.write_text(json.dumps({"answer": "just one record"}), encoding="utf-8")

        documents = load(path)

        assert len(documents) == 1
        assert documents[0].metadata["locator"] == "record 1"

    def test_invalid_json_is_an_input_error(self, tmp_path: Path):
        path = tmp_path / "broken.json"
        path.write_text('{"unterminated', encoding="utf-8")

        with pytest.raises(InputError, match="not valid JSON"):
            load(path)

    def test_no_readable_records_is_an_input_error(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")

        with pytest.raises(InputError, match="no readable records"):
            load(path)


class TestLoadPdf:
    def test_pdf_without_text_is_an_input_error(self, tiny_pdf: Path):
        """Blank pages stand in for a scan: no extractable text, so say so."""
        with pytest.raises(InputError, match="no extractable text"):
            load(tiny_pdf)

    def test_corrupt_pdf_is_an_input_error(self):
        with pytest.raises(InputError, match="could not read the PDF"):
            load_bytes(b"%PDF-1.4 truncated nonsense", "broken.pdf")


class TestFileTypes:
    def test_unsupported_extension_from_disk(self, tmp_path: Path):
        path = tmp_path / "notes.txt"
        path.write_text("plain text", encoding="utf-8")

        with pytest.raises(InputError, match="unsupported file type"):
            load(path)

    def test_unsupported_extension_from_upload(self):
        with pytest.raises(InputError, match="unsupported file type"):
            load_bytes(b"plain text", "notes.txt")

    def test_upload_and_disk_paths_agree(self, records_json: Path):
        """One set of parsers, so the API and the CLI cannot drift apart."""
        from_disk = load(records_json)
        from_upload = load_bytes(records_json.read_bytes(), records_json.name)

        assert [d.page_content for d in from_disk] == [d.page_content for d in from_upload]
        assert [d.metadata for d in from_disk] == [d.metadata for d in from_upload]
