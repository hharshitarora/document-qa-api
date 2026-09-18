# Build log

What was observed and measured while building, in order. Decisions and their reasons live in `decisions.md`; this file is the evidence behind them.

## Setup
- Sample PDF: 84 pages, all with extractable text, so no OCR path is needed.
- A gpt-4o-mini test call on the provided key returned 200. Budget is capped at $5.
- The sample "JSON" file in the brief is a spreadsheet of already-answered questions with `id, question, answer, comments, confidence` columns, and `Data-Not-Found` where the document had no answer. That is the answer shape they think in.

## Ingest and index
- Raw pypdf text: 259,445 characters. Collapsing runs of whitespace: 199,707. **23% of the extracted text was layout padding.**
- Before header stripping: 291 chunks at 1200/200, average 1028 characters.
- After whitespace normalisation: 232 chunks. After header stripping: 228 chunks.
- Embedding 228 chunks with `text-embedding-3-small` takes about 4 seconds and costs a fraction of a cent. Load and split takes about 5 seconds, most of it pypdf text extraction.

### Running header
- The same title line appears on all 84 pages and was present in 84 of 232 chunks.
- **First attempt failed.** Detecting boilerplate by counting repeated lines deleted real content: pypdf emits many single words on their own lines in this document, so common words like "audit", "of" and "place" crossed the repetition threshold. Total text dropped from 199k to 136k characters and body sentences came out with words missing.
- **Second attempt.** Detect the longest text prefix shared by at least 30% of pages. A fixed-step scan matched mid-word and left every page starting with "ity", so the match is grown to the full shared prefix and cut back to a word boundary.

### Retrieval check
Question: "Which cloud providers do you rely on?" Scores from `similarity_search_with_score`, k=20.
- All of the top 5 chunks mention GCP, the provider the report actually names.
- 17 chunks in the corpus mention GCP; the AWS and Azure mentions on page 14, which are background rather than what Nave uses, rank 18th and fall outside the cut.
- Judging retrieval from the first 90 characters of each chunk was misleading: the chunks looked generic while in fact containing the answer. Worth checking full text before concluding anything about retrieval.

Question with no answer in the document: "What is the CEO's home address?"
- Best score 0.289 against 0.401 for the real question. **The gap is thin**, which is evidence that a fixed similarity threshold for "not found" would be fragile and that refusal belongs in the answering step instead.
