# Document Q&A

Answers questions about a document you upload, using only what that document says. Every
answer carries the page or record it came from, with the sentence it was taken from, and
questions the document does not cover come back as `Data-Not-Found` rather than a guess.

PDF and JSON documents are supported. Questions arrive as a JSON file, or one at a time.

```bash
curl -X POST http://localhost:8000/qa \
  -F document=@samples/nave-soc2-type2.pdf \
  -F questions=@samples/questions.json
```

```json
{
  "document": "nave-soc2-type2.pdf",
  "chunks": 228,
  "cached": false,
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "found": true,
      "answer": "The cloud providers relied on are Google Cloud Platform (GCP) for hosting and infrastructure services.",
      "citations": [{"location": "page 17", "snippet": "GCP Cloud hosting provider"}],
      "error": null
    },
    {
      "question": "Which of the following, if any, are performed as part of your monitoring process for the service: Application Performance Monitoring (APM), End User Monitoring (EUM), Digital Experience Monitoring (DEM)",
      "found": false,
      "answer": "Data-Not-Found",
      "citations": [],
      "error": null
    }
  ]
}
```

## Run it

With Docker, which needs only an API key:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
docker compose up --build
```

Or locally, with Python 3.11:

```bash
python -m venv .venv && .venv/Scripts/activate     # source .venv/bin/activate on macOS and Linux
pip install -r requirements.txt
cp .env.example .env                                # then add your key
uvicorn app.api:app --reload
```

Then open **http://localhost:8000** for the upload page, or **/docs** for the API explorer.

There is also a command-line runner, which is the quickest way to try the pipeline
without the service:

```bash
python scripts/ask.py --doc samples/nave-soc2-type2.pdf --questions samples/questions.json
python scripts/ask.py --doc samples/company-kb.json --question "Where are your data centres located?"
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /qa` | `document` (.pdf or .json) plus either a `questions` file or a single `question` field |
| `GET /health` | status, models in use, and the active limits |
| `GET /` | minimal upload page |
| `GET /docs` | generated API explorer |

Limits, all overridable by environment variable: 20MB per upload, 50 questions per
request, 5 questions answered concurrently, 30 second timeout per model call.

Errors carry a message and the right status: `400` for anything wrong with the upload or
the questions, `413` when a file is too large, `502` when the model or embedding call
fails. A single question failing does not fail the batch: that result carries an `error`
and the rest are answered normally.

## How it works

Once per document: read it (one Document per page or per record), strip the running
header and layout whitespace, split into 1200-character chunks with 200 overlap carrying
the page or record number, embed every chunk, hold the vectors in memory.

Once per question: embed the question, take the 5 closest chunks, show them to
gpt-4o-mini labelled `[page 17]`, and require a structured answer with `found`, `answer`
and `citations`.

Then the part that matters: **the model's answer is checked before it is returned.**
Citations naming a passage that was never retrieved are dropped as fabricated, quotes
that are not verbatim lose the quote but keep the location, and an answer with no
surviving citation is downgraded to `Data-Not-Found`. `found` is recomputed by the
service, never taken from the model.

`flow.md` has the step-by-step version, including where untrusted model output becomes a
checked result.

## Design decisions

The full log is in `decisions.md`, 41 entries with the alternative rejected and the
tradeoff accepted. The five that shaped the most:

**LangChain for the commodity parts, the rest by hand.** Splitting, embeddings and the
vector store interface come from LangChain. Retrieval, the prompt and the answer contract
are written here, because those decide whether an answer can be trusted. A prebuilt
retrieval chain would have been faster to assemble and would have owned the decisions
this project is actually about.

**Vectors in memory, behind the vector store interface.** One document is a few hundred
chunks, so exact search is instant and needs no extra service; Chroma or pgvector is a
change in one file. The cost is that nothing survives a restart.

**Split page by page, never joining pages.** A citation is only as precise as the unit
kept, so pages are never merged and every chunk can name its page. The cost is that a
control spanning a page break loses its overlap.

**Citations verified against what was actually retrieved.** The alternative, trusting the
model's own citations, is the difference between claiming grounding and proving it. This
is stricter than the brief asks: an answer whose quote was paraphrased loses the quote,
and one whose citations all fail is reported as not found.

**The document cache is checked before parsing, not after.** Caching the embeddings alone
saved nothing measurable, because parsing the PDF is the slower half. Hashing the bytes
first took a repeat request from 12.3s to 2.4s.

## Grounding, and how it is measured

Two evals, deliberately separate, because a wrong answer and a missing passage are
different problems and were confused for each other twice during this build.

**`python scripts/eval_retrieval.py`** asserts that the passage holding each answer
reaches the top 5. Embeddings only, no chat calls. Currently **recall@5 of 6/8**.

**`python scripts/eval.py`** asserts that each question's found or not-found decision is
right and that any answer carries a citation. Currently **9/11**.

Both failures are the same two questions, which is the point: they are retrieval misses,
not answering problems.

- *Who signed the report?* The signature block on page 7 is a very short chunk and never
  reaches the top 60 for any phrasing.
- *Incident notification criteria and SLAs.* Page 21 says "Nave will inform all necessary
  parties of the incident without undue delay", which supports a partial answer. It ranks
  36th, buried under the control-testing matrix that makes up a third of the report.

An independent review by a different model found a third problem, since fixed: the
service was answering "Yes, personal information is transmitted to third parties" from a
passage describing a vendor risk assessment control. The citation was accurate; the
conclusion was not supported. The prompt now separates the two, so a passage about a
policy or an auditor's test proves the control exists and not that the activity happens,
and two eval cases hold that line. For a questionnaire tool this failure mattered more
than a refusal would have: a confident yes gets pasted into a real security assessment.

### About the sample files

The brief's five sample questions, run against both sample files:

| Question | Sample PDF | Sample JSON |
|---|---|---|
| 1. Notification criteria and SLAs | `Data-Not-Found` | `Data-Not-Found` |
| 2. Personal information to third parties | `Data-Not-Found` | `Data-Not-Found` |
| 3. Cloud providers | answered, page 17 | answered, record 1 |
| 4. Data centre region and backups | `Data-Not-Found` | answered, record 1 |
| 5. APM, EUM, DEM | `Data-Not-Found` | `Data-Not-Found` |

Most of those refusals are correct, and one is not:

- **Question 1 is our miss.** Page 21 of the PDF says "Nave will inform all necessary
  parties of the incident without undue delay", which supports a partial answer. It ranks
  36th, so it never reaches the model. This is a retrieval limitation, described below.
- **Question 2** is correct on both. The PDF describes vendor risk controls and never
  states that personal information flows to vendors, and confidentiality and privacy are
  outside the report's scope. No record in the JSON file covers it.
- **Questions 4 and 5** have no answer in the PDF at all: "region" and "APM" each appear
  0 times across its 84 pages. Question 4 is answered from the JSON file.

Worth knowing why the PDF covers so little of its own appendix: the sample answers
supplied with the questions contain facts such as "US Central region" that appear nowhere
in that PDF, and cite an internal knowledge base file as their source. The two sample
files are not a matched pair. `samples/company-kb.json` is that answer spreadsheet
converted to JSON, which is why several questions answer from it and not from the report.

## Security and data handling

Uploads are parsed in memory and never written to disk, because document text here is
customer compliance data. Nothing about a document is persisted: the index lives in
process memory, keyed by a hash of the bytes, bounded to 8 documents, and gone on restart.

Logs are JSON, one line per request, carrying counts, timings, token usage and estimated
cost, and never document text, question text or key material. Each response carries an
`x-request-id` header matching its log lines, so a reported problem can be traced without
logging what was asked.

The key is supplied at run time and is excluded from the image by `.dockerignore`. The
container runs as a non-root user.

## Tests

```bash
pytest
```

61 tests, offline, about a second: no network, no model calls, no cost. They cover
loading and cleaning, chunking and metadata, JSON flattening, every error path, the
endpoint including each validation failure, and the cache, which is checked by counting
parses rather than timing them.

The highest-value ones are on verification: a citation naming a page that was never
retrieved is dropped, a paraphrased quote keeps its location and loses the snippet, a
quote spanning two chunks of one page still verifies, and a claimed answer with no
surviving citation becomes not found.

Answer quality is not in this suite on purpose. Judging it needs real model calls, so it
lives in the two eval scripts above.

## Limitations, and what I would do next

**Chunking, and it is the first thing I would change.** Both retrieval misses are chunking
problems wearing different clothes: a short fragment that starves for context, and a long
chunk judged on its dominant topic while the answer sits in its final 15 characters.
Splitting on section headings such as "5.12 Incident Management" would keep a section
together and let it be judged as itself.

**Not a ranking problem, and I checked.** A reranking stage was built and measured: fetch
40 candidates, have gpt-4o-mini pick 5. Results were 9/11 with it and 9/11 without, no
case improved, at one extra model call and about 10,000 extra input tokens per question.
It was reverted. The experiment is written up in `build-log.md`.

**Tables are flattened.** pypdf gives text, not structure, so a table becomes a line of
values. The maintained parsers that do better each cost something: PyMuPDF is AGPL,
Docling downloads models, Unstructured needs system packages.

**Upload size is checked after the body is in memory**, not during the read. Enforcing it
while receiving needs a streaming request body.

**The cache is per process**, so it does not help across replicas. Redis or a shared
vector store would.

**No auth and no rate limiting.** Both belong at the edge in front of this service, and
neither was in scope.

**A faithfulness judge would be the next eval**, checking that a cited passage actually
supports the claim made from it. That is the check that would have caught the
control-versus-fact error automatically instead of an external review finding it.

## AI tool usage

I spent time going through the brief first and building a rough plan in my head for how I
wanted to approach it, then discussed that with Claude Code and split the work into
phases. We went through each phase in depth before any code was written.

While working through the phases I also read up on the specifics myself: the RAG material
they linked, how to test outputs, how evals are done, and LangChain implementation
details.

Then we went phase by phase. At each phase I checkpointed, tested, and looked at whether
the output quality and the direction were right. Where I felt the quality or the reasoning
was heading the wrong way, we worked through it before moving on. The decisions, the
tradeoffs and the calls on what to cut were mine; the typing was largely Claude's.

`decisions.md` and `build-log.md` are the running record of that, kept as the work
happened rather than written afterwards, including the experiments that failed.

## Time spent

Roughly 4 to 5 hours, including the research, the evals, and two experiments that were
measured and then reverted.
