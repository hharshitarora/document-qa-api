"""Ask one question about one document, from the command line.

    python scripts/ask.py --doc samples/nave-soc2-type2.pdf --question "Which cloud providers do you rely on?"

The document is re-read and re-embedded on every run, which costs a few seconds and a
fraction of a cent. Caching belongs with the API, where the same document is uploaded
repeatedly.
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running this file directly, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.index import build_index, search  # noqa: E402
from app.ingest import load_pdf, split  # noqa: E402
from app.qa import answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a PDF.")
    parser.add_argument("--doc", type=Path, required=True, help="path to a PDF")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    chunks = split(load_pdf(args.doc))
    store = build_index(chunks)
    indexed = time.perf_counter() - started

    hits = search(store, args.question)
    text, usage = answer(args.question, hits)

    print(f"\nQ: {args.question}")
    print(f"A: {text}\n")
    print(f"pages retrieved: {[doc.metadata['page'] for doc, _ in hits]}")
    print(f"chunks: {len(chunks)}  index: {indexed:.1f}s  total: {time.perf_counter() - started:.1f}s")
    print(f"tokens: {usage.get('input_tokens')} in, {usage.get('output_tokens')} out")


if __name__ == "__main__":
    # The Windows console defaults to cp1252 and the document contains characters it
    # cannot encode.
    sys.stdout.reconfigure(encoding="utf-8")
    main()
