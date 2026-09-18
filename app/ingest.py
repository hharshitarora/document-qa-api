"""Turn a source document into chunks ready for embedding.

pypdf is used directly rather than through a LangChain loader: the loaders for PDF
live in langchain-community, which is being sunset, and the maintained alternatives
each carry a cost (AGPL licensing, model downloads, or system packages) that is not
worth paying to read one text-based PDF. Reading pages ourselves is a few lines and
keeps control over the metadata that citations depend on.
"""

import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# A running header repeats identically at the start of most pages, so it adds no
# information and pulls every chunk towards the same generic meaning. It is detected as
# a text prefix shared by at least this share of pages, and only if long enough to be a
# header rather than a coincidence.
HEADER_PAGE_SHARE = 0.3
HEADER_MIN_CHARS = 20
HEADER_MAX_CHARS = 200

# A self-contained unit in a compliance report is one control description, a paragraph
# or two. 1200 characters (roughly 300 tokens) sits just above that, so a control
# survives in one chunk. Overlap is ~15%, insurance against a split landing
# mid-sentence. See decisions.md.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces.

    pypdf preserves the visual layout, so words arrive separated by two or three
    spaces. Measured on the sample report that is 23% of all characters, which means
    chunks hold a quarter less real content and every retrieved chunk spends tokens
    on padding. Layout is lost, but this parser already flattens tables either way.
    """
    return re.sub(r"\s+", " ", text).strip()


def _find_running_header(page_texts: list[str]) -> str:
    """The longest text prefix shared by a large share of pages, or "".

    Detected rather than hardcoded, so it works on any document. Longest first, so the
    whole header is found rather than its first few words.

    Line-frequency detection was tried first and had to be abandoned: this parser emits
    many single words on their own lines, so common words like "audit" and "place" were
    counted as boilerplate and deleted from the body text.
    """
    if len(page_texts) < 3:
        return ""

    threshold = max(2, int(len(page_texts) * HEADER_PAGE_SHARE))
    for length in range(HEADER_MAX_CHARS, HEADER_MIN_CHARS - 1, -5):
        candidates = Counter(text[:length] for text in page_texts if len(text) > length)
        prefix, count = candidates.most_common(1)[0] if candidates else ("", 0)
        if count < threshold:
            continue

        # The scan steps in fixed jumps, so the match can land mid-word. Grow it to the
        # full shared prefix, then cut back to a word boundary: stripping "...Relevant
        # To Secur" would otherwise leave every page starting with "ity".
        matching = [text for text in page_texts if text.startswith(prefix)]
        shared = matching[0]
        for text in matching[1:]:
            limit = min(len(shared), len(text))
            cut = limit
            for position in range(limit):
                if shared[position] != text[position]:
                    cut = position
                    break
            shared = shared[:cut]

        return shared[: shared.rfind(" ") + 1].strip() if " " in shared else shared
    return ""


def load_pdf(path: Path) -> list[Document]:
    """One Document per page, with a 1-based page number for citations.

    Pages that hold no extractable text (scans, image-only pages) are skipped:
    they would embed to noise and could be retrieved in place of real content.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))

    # Two passes: the first to learn the running header, the second to build the
    # Documents without it.
    page_texts = [_normalise_whitespace(page.extract_text() or "") for page in reader.pages]
    header = _find_running_header(page_texts)

    documents: list[Document] = []
    for number, page_text in enumerate(page_texts, start=1):
        text = page_text[len(header):].strip() if header and page_text.startswith(header) else page_text
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(path).name,
                    "page": number,
                    # One label for both file types, so citations read the same whether
                    # the source has pages or records. See decisions.md.
                    "locator": f"page {number}",
                },
            )
        )

    return documents


def _flatten(record: object) -> str:
    """A JSON value as readable text.

    Objects become `key: value` lines, which keeps the field names in the embedded
    text: a question asking about data centres should match a record whose key is
    "answer" and whose value mentions them. Nested values are dumped back to JSON
    rather than dropped, so nothing in the file is silently lost.
    """
    if isinstance(record, dict):
        lines = []
        for key, value in record.items():
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if str(text).strip():
                lines.append(f"{key}: {text}")
        return "\n".join(lines)
    if isinstance(record, list):
        return "\n".join(_flatten(item) for item in record)
    return "" if record is None else str(record)


def load_json(path: Path) -> list[Document]:
    """One Document per record, with a record number for citations.

    No assumption is made about the shape: the brief promises only that the API
    accepts JSON. A list becomes one Document per item, a single object becomes one
    Document. See decisions.md.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else [data]

    documents: list[Document] = []
    for number, record in enumerate(records, start=1):
        text = _normalise_whitespace(_flatten(record)) if not isinstance(record, dict) else _flatten(record)
        if not text.strip():
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(path).name,
                    "record": number,
                    "locator": f"record {number}",
                },
            )
        )

    return documents


def load(path: Path) -> list[Document]:
    """Load a document by file type. The only place file types are decided."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".json":
        return load_json(path)
    raise ValueError(f"unsupported file type '{suffix}': expected .pdf or .json")


def split(documents: list[Document]) -> list[Document]:
    """Split each Document into overlapping chunks, carrying its metadata along.

    Splitting is per page and pages are never joined, so every chunk can name the
    page it came from. The cost is that a control spanning a page break is split
    with no overlap to rescue it. JSON records are usually short enough to stay whole.
    See decisions.md.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)
