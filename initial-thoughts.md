# Initial thoughts

Written before any pipeline code, kept as-is so the reasoning is visible.

## The problem in one line
Given a document and a list of questions, answer each question using only what the document says.

## How I would phase it from the bare problem statement
Ordered by risk, not by component. The thinnest thing that works end to end comes first, then it gets widened. Building ingest fully, then splitting fully, then retrieval fully means finding out whether the whole thing holds together at the very end, which is the worst time to find out.

1. **Walking skeleton.** One document, one question, an answer printed. Hardcoded and ugly. This kills every unknown at once: can I read the file, do embeddings work, does the model answer from what I gave it. Everything after this is improvement rather than discovery.
2. **Make the answer trustworthy.** Cite what it used, refuse when the document does not cover the question. This is the actual product, and steps 1 and 2 are where the thinking lives.
3. **Make it general.** Many questions, more than one file format, no hardcoded paths.
4. **Make it reachable.** An interface so something other than me can call it.
5. **Make it survive bad input.** Wrong file type, oversized file, model failure, nonsense questions.
6. **Make it provable.** Tests, and a way of checking answer quality that is not me reading them.
7. **Make it runnable by a stranger.** Packaging and documentation.

Rule holding it together: every phase ends with something runnable. If a phase ends with code that only makes sense once the next phase exists, the split was wrong.

## What scoping the brief changed
The order did not change. What changed was how much to invest at each end, and that is the point of scoping before building:
- The rubric put real weight on robustness, tests, containerization and observability, so phases 5 to 7 get proper time rather than a token pass.
- A minimal UI is worth points, so it is in scope instead of being cut.
- Answers must be grounded with citations and must return a not-found result when unsupported, which confirms step 2 is the core rather than a nicety.
- Their own sample answer file shows the shape they think in: an answer, a supporting comment, a confidence level, and `Data-Not-Found` where the document says nothing.
- Model and budget are fixed (gpt-4o-mini, a $5 cap), which rules out anything that spends tokens casually.

## Why retrieval instead of sending the whole document
Their sample PDF is 84 pages, roughly 65,000 tokens. Sending all of it per question would half work, but it costs more, runs slower, and accuracy drops when the answer is buried in that much text. Retrieval means each question only sees the passages that matter.

## Where the framework ends and my code starts
LangChain owns the commodity steps: loading files, splitting text, embedding, storing vectors, similarity search. Retrieval strategy, the prompt, the answer contract and the not-found rule are written by hand, because those are the parts that decide answer quality and the parts worth defending.

## Open questions I am carrying into the build
- What is the unit of source: a page for PDFs, a record for JSON, and what metadata travels with it.
- How big a chunk should be, and how much overlap, for a document this structured.
- How many chunks per question, and whether a similarity floor is needed before calling something not found.
- Whether to verify grounding with a second pass, and whether it earns its extra call.
