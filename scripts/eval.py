"""Answer-quality check against a real document and real model calls.

Deliberately not part of pytest: the suite must run offline and free, and this spends
tokens. Run it by hand when the prompt, the chunking or the retrieval settings change.

    python scripts/eval.py

Each case says what the document supports, not what words the answer should use. A case
passes when the system's found/not-found decision matches, and any answer it gives is
backed by a citation.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.index import build_index  # noqa: E402
from app.ingest import load, split  # noqa: E402
from app.qa import answer_all  # noqa: E402

PDF = Path("samples/nave-soc2-type2.pdf")
KB = Path("samples/company-kb.json")

# (document, question, should the document answer it, why)
CASES = [
    (PDF, "Which cloud providers do you rely on?", True, "the report names GCP as its hosting provider"),
    (PDF, "Is personal information disclosed to third parties?", True, "vendor management and third-party review are described"),
    (PDF, "Who signed the report?", True, "the signature block names the CEO"),
    (PDF, "What is the CEO's home address?", False, "nowhere in the report"),
    (PDF, "What is the company's revenue?", False, "a SOC 2 report carries no financials"),
    (PDF, "Which APM tool is used: Datadog or New Relic?", False, "no APM product is named anywhere"),
    (KB, "Where are your data centres located?", True, "record 1 says US Central on GCP"),
    (KB, "Do you have a dedicated sanctions compliance officer?", True, "answered as No, with a reason"),
    (KB, "How many employees do you have?", False, "no record covers headcount"),
]


async def main() -> None:
    passed = failed = 0

    for document_path in (PDF, KB):
        cases = [case for case in CASES if case[0] == document_path]
        if not cases:
            continue

        store = build_index(split(load(document_path)))
        results, usage = await answer_all([question for _doc, question, _expected, _why in cases], store)

        print(f"\n{document_path.name}")
        for (_doc, question, expected, why), result in zip(cases, results):
            ok = result.found == expected
            backed = (not result.found) or bool(result.citations)
            if ok and backed:
                passed += 1
                mark = "pass"
            else:
                failed += 1
                mark = "FAIL"

            print(f"  [{mark}] expected {'an answer' if expected else 'not found'}: {question}")
            print(f"         {why}")
            print(f"         got: {result.answer[:120]}")
            if result.citations:
                print(f"         cited: {', '.join(c.location for c in result.citations)}")
            if result.error:
                print(f"         error: {result.error}")
            if result.found and not backed:
                print("         FAIL reason: answered with no citation")

        print(f"  tokens: {usage['input_tokens']} in, {usage['output_tokens']} out")

    total = passed + failed
    print(f"\n{passed}/{total} cases behaved as expected")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
