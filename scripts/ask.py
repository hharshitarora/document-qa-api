"""Ask questions about a document from the command line.

    python scripts/ask.py --doc samples/nave-soc2-type2.pdf --questions samples/questions.json
    python scripts/ask.py --doc samples/company-kb.json --questions samples/questions.json
    python scripts/ask.py --doc samples/nave-soc2-type2.pdf --question "Which cloud providers do you rely on?"

The document is re-read and re-embedded on every run, which costs a few seconds and a
fraction of a cent. Caching belongs with the API, where the same document is uploaded
repeatedly.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Allow running this file directly, not only as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.index import build_index  # noqa: E402
from app.ingest import load, split  # noqa: E402
from app.qa import answer_all  # noqa: E402


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("--doc", type=Path, required=True, help="path to a .pdf or .json document")
    parser.add_argument("--questions", type=Path, help="JSON file holding a list of questions")
    parser.add_argument("--question", help="a single question")
    args = parser.parse_args()

    if not args.questions and not args.question:
        parser.error("pass --questions or --question")

    questions: list[str] = (
        json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else []
    )
    if args.question:
        questions.append(args.question)

    started = time.perf_counter()
    chunks = split(load(args.doc))
    store = build_index(chunks)
    print(f"{len(chunks)} chunks indexed in {time.perf_counter() - started:.1f}s\n")

    results, usage = await answer_all(questions, store)

    for number, result in enumerate(results, start=1):
        print(f"{number}. {result.question}")
        print(f"   {result.answer}")
        if result.error:
            print(f"   error: {result.error}")
        for citation in result.citations:
            print(f"   {citation.location}: {citation.snippet or '(no exact quote returned)'}")
        print()

    print(
        f"{len(results)} questions, {usage['input_tokens']} tokens in, "
        f"{usage['output_tokens']} out, {time.perf_counter() - started:.1f}s total"
    )


if __name__ == "__main__":
    # The Windows console defaults to cp1252 and the document contains characters it
    # cannot encode.
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
