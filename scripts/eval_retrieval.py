"""Retrieval check: is the passage that holds the answer actually retrieved?

Separate from `eval.py` on purpose. That script grades what the system answered, which
mixes two very different failures together. This one grades only retrieval, using
embeddings and no chat calls, so it is fast, cheap and deterministic.

    python scripts/eval_retrieval.py

The distinction matters: if the right passage was retrieved and the answer is still
wrong, that is a prompt problem. If it was never retrieved, no prompt change can help.
Both were confused for each other during this build until they were measured apart.

When a case misses, the rank the passage did reach is printed, which is the diagnosis
rather than just a failure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.index import TOP_K, build_index, search  # noqa: E402
from app.ingest import load, split  # noqa: E402

PDF = Path("samples/nave-soc2-type2.pdf")
KB = Path("samples/company-kb.json")

# How deep to look when a case misses, so the report can say how far off it was.
DIAGNOSTIC_K = 60

# (document, question, locator holding the answer, how that was confirmed)
CASES = [
    (PDF, "Which cloud providers do you rely on?", "page 17",
     "the Third-Party Summary table names GCP as the cloud hosting provider"),
    (PDF, "Who signed the report?", "page 7",
     "the signature block names the CEO"),
    (PDF, "Do you have formally defined criteria for notifying a client during an incident? What are your SLAs?", "page 21",
     "'Nave will inform all necessary parties of the incident without undue delay'"),
    (PDF, "How is data encrypted in transit?", "page 67",
     "SQL databases, Redis, Memcache and cloud storage buckets are described as encrypted"),
    (PDF, "What data does Nave classify as confidential?", "page 22",
     "the confidential data list, including PII and customer data"),
    (KB, "Where are your data centres located?", "record 1",
     "US Central on GCP"),
    (KB, "Do you have a dedicated sanctions compliance officer?", "record 12",
     "answered No, with a compliance programme described"),
    (KB, "Do you monitor unauthorised software installation?", "record 2",
     "application whitelisting and blacklisting"),
]


def main() -> None:
    passed = failed = 0

    for document_path in (PDF, KB):
        cases = [case for case in CASES if case[0] == document_path]
        if not cases:
            continue

        chunks = split(load(document_path))
        store = build_index(chunks)
        print(f"\n{document_path.name}  ({len(chunks)} chunks)")

        for _doc, question, expected, why in cases:
            hits = search(store, question, k=DIAGNOSTIC_K)
            locators = [doc.metadata["locator"] for doc, _score in hits]
            rank = locators.index(expected) + 1 if expected in locators else None

            if rank is not None and rank <= TOP_K:
                passed += 1
                print(f"  [pass] {expected} at rank {rank}: {question[:62]}")
            else:
                failed += 1
                where = f"rank {rank}" if rank else f"outside the top {DIAGNOSTIC_K}"
                print(f"  [MISS] {expected} at {where}, needed top {TOP_K}: {question[:62]}")
                print(f"         expected because {why}")
                print(f"         top {TOP_K} were: {', '.join(locators[:TOP_K])}")

    total = passed + failed
    print(f"\nrecall@{TOP_K}: {passed}/{total}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
