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

## First end-to-end answers
Command: `python scripts/ask.py --doc samples/nave-soc2-type2.pdf --question "..."`

Answerable question, "Which cloud providers do you rely on?":
- Answer: "The cloud providers relied on are GCP (Google Cloud Platform) for cloud hosting and infrastructure services. Data-Not-Found"
- Pages retrieved 17, 16, 65, 81, 81. 943 input tokens, 25 output. About 11 seconds, 9 of them re-embedding the document.

Unanswerable question, "What is the CEO's home address?":
- Answer: `Data-Not-Found`, 891 input tokens, 5 output.
- Page 7 was retrieved, the signature block naming the CEO. So retrieval landed near the question and the model still declined, which is the behaviour wanted.

**Defect found, deferred to Phase 4.** The answerable question returned a correct answer *and* appended `Data-Not-Found`. The prompt says to use that string when the passages do not contain the answer, and the model read it as something to add rather than something to replace. Any parser reading that result would treat a good answer as a refusal. A one-sentence prompt patch would hide it; the real fix is a structured response where the answer and the found/not-found state are separate fields, so the contradiction cannot be expressed. Left in place deliberately at the end of Phase 3.

## Structured answers, and what the sample PDF actually covers
The Phase 3 defect is gone: the model now fills `found`, `answer` and `citations` as separate fields, so an answer can no longer carry a refusal string inside it.

Run across all 5 appendix questions plus a planted unanswerable one, 6259 input tokens, 97 output, 19.6s including a 10s index build:
- Question 3, cloud providers: answered, "GCP (Google Cloud Platform) is the cloud provider relied on for hosting and infrastructure", cited page 17 with an exact quote that passed verification.
- Questions 1, 2, 4, 5 and the planted one: `Data-Not-Found`.

**Verification was not the cause.** Instrumented the model's raw output against the verified output: the model itself set `found` to false on all four, with no citations offered. So the refusals come from the passages, not from our checking.

**The sample PDF does not contain those answers.** Term counts across all 84 pages:
- "region": 0 occurrences, so question 4 (data centre region) cannot be answered.
- "APM": 0, so question 5 (APM, EUM, DEM by name) cannot be answered.
- "personal information": 0, so question 2 as phrased cannot be answered.
- Notification: the document lists a "Breach Notification Policy" by name and states no criteria or timeframes. Pulled the closest passages by hand at k=20; the best are about vendor relationships at 0.44. There is no answer to find for question 1.

Their own spreadsheet answers contain facts like "US Central region" that appear nowhere in the PDF, and cite an internal knowledge base file as their source. So the two sample files in the appendix are not a matched pair: the questions were written against their knowledge base, not against the Nave report. The refusals are correct behaviour, but four not-founds could read as a broken app, so the README has to carry this evidence.

## JSON path, and two model-behaviour findings
JSON loading added, one Document per record, citations labelled `record N`. 19 records become 19 chunks, indexed in under a second.

**Over-refusal on a borderline question.** "Which cloud providers do you rely on?" retrieved record 1 at rank 1 with the run's best score, 0.48, and that record plainly says "hosted within Google Cloud Platform (GCP)". The model still set `found` to false. Rewording the question showed why:
- "Which cloud providers do you rely on?" → refused
- "Which cloud provider hosts your infrastructure?" → answered, GCP
- "Do you use Google Cloud Platform?" → answered, yes

So the plural is read as a request for the complete list of providers, and the model declines rather than imply GCP is all of them. Careful behaviour, not a fault, but it exposed that only two outcomes existed: a full answer or nothing.

**Partial answers needed an example, not an instruction.** Three prompt variants, same context:
- Strict rule: refused.
- "Answer whatever the passages support, even if only part": still refused.
- Same, plus one worked example: answered, and still refused the planted unanswerable question.

**Prose beats bullets.** The winning wording, moved into the app as a bulleted list under a "Rules:" heading, went back to refusing. Removing the citation rule, the records sentence, and the "leave answer empty" clause one at a time did not change it, twice each at temperature 0. As prose, it answers. Same instructions, different layout, different behaviour on borderline questions.

## Coverage of their 5 questions, both files
- Q2 third parties: answered from the PDF, page 45, quote verified.
- Q3 cloud providers: answered from the PDF, page 17, quote verified.
- Q4 data centre region: answered from the JSON, record 1, partially, naming the missing backup locations.
- Q5 APM, EUM, DEM: those acronyms appear 0 times in either file. Genuinely unanswerable, though a partial answer about anomaly monitoring would be possible.
- Q1 notification criteria and SLAs: **a retrieval miss, not a refusal.** The PDF names a "Breach Notification Policy" on page 20, and no chunk containing the word "notification" appears in the top 25 for that question. Tested at k=5, 10, 15 and 20: `Data-Not-Found` at every depth. Page 20 is a bare list of policy titles and the question is a long two-part sentence, so they sit far apart in embedding space regardless of depth. Keyword search finds it instantly, which is why hybrid retrieval is the right fix and why raising k is not.

## API
`uvicorn app.api:app`, then real uploads rather than a test client:
- `GET /health` returns status and both model names, no key material.
- `POST /qa` with their PDF and their questions file: 228 chunks, 5 results, questions 2 and 3 answered with verified citations on pages 45 and 17. Same output as the command line, which is the point of sharing one set of parsers.
- `POST /qa` with `company-kb.json`: 19 chunks, question 4 answered partially from record 1.
- `POST /qa` with a `.txt` file as the document: **HTTP 500**. The loader raises `ValueError`, nothing catches it. Correct diagnosis, wrong status code, and the fix belongs in the error-handling phase rather than being patched inline here.

## Robustness and concurrency
Validation, measured against the running service rather than a test client:

| Request | Result |
|---|---|
| `.txt` as the document | 400 `unsupported file type '.txt': expected .pdf or .json` |
| truncated JSON document | 400 `document is not valid JSON: Unterminated string ...` |
| truncated questions file | 400 `questions file is not valid JSON: ...` |
| 51 questions | 400 `51 questions exceeds the limit of 50 per request` |
| neither questions file nor question field | 400 `send a questions file, or a "question" field` |
| single `question` form field, no file | 200, answered from page 17 |

**The caching mistake, worth keeping.** The first implementation cached the built index keyed by a hash of the document, but the lookup sat *after* parsing and splitting. Measured cold 11.0s, repeat 11.7s: no gain, because pypdf parsing is the slower half of preparing the sample report. Moving the hash check ahead of parsing, into `app/pipeline.py`, took a repeat request from 12.3s to 2.4s. Caching the expensive step is not the same as caching early enough to skip it.

**Concurrency.** 20 questions against a cached document: 7.1s total, 0.36s per question, no errors, at 5 in flight. Sequential answering earlier measured about 1.2s per question, so roughly a 3x improvement, bounded by the semaphore rather than by the API.

## Tests
61 tests, offline, about 1.2 seconds: ingest and cleaning (23), verification and answering (15), the endpoint (18), the cache (5).

- The fake chat model has to imitate `with_structured_output(..., include_raw=True)`, which returns `{parsed, raw, parsing_error}` rather than text.
- Two header tests failed first time and both were bad fixtures, not bad code: giving every page identical body text made the shared prefix run past the header into the body. Realistic fixtures with differing bodies pass.
- One test found a real gap: a questions file shaped `{"nope": 1}` fell through to "no questions were provided", which does not say what is wrong. Now it says the object needs a `"questions"` key.
- The cache tests count parses and index builds rather than timing anything, so they prove the repeat request skips both.

## Answer quality, `scripts/eval.py`
Nine cases across both files, each asserting the found/not-found decision and that any answer carries a citation. **7 of 9.** 10,065 input tokens, 272 output.

Both failures are on the PDF:
- "Is personal information disclosed to third parties?" returns not found, although the longer original wording of the same question answers from page 45. Phrasing changes retrieval.
- "Who signed the report?" returns not found, although page 7 holds the signature block naming the CEO. The model will not infer that a signature answers "who signed".

Same root cause as the notification miss: dense-only retrieval plus a literal model. Left failing on purpose, with the numbers published.

## Sample JSON provenance
Their "Sample JSON file" link is a spreadsheet, not JSON. Exported to CSV, then converted with `csv.DictReader` plus `json.dumps` into `samples/company-kb.json`: 19 records with `id, question, answer, comments, confidence`, dropping the unnamed export index column. Done as a one-off, no script kept.

Question with no answer in the document: "What is the CEO's home address?"
- Best score 0.289 against 0.401 for the real question. **The gap is thin**, which is evidence that a fixed similarity threshold for "not found" would be fragile and that refusal belongs in the answering step instead.
