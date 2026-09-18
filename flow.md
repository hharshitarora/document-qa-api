# Flow

One page, end to end. Objects in `app/`, reasons in `decisions.md`, measurements in `build-log.md`.

## Once per document

| Step | Code | In | Out |
|---|---|---|---|
| 1. Load | `ingest.load_pdf` | PDF path | one `Document` per page: text plus `{source, page}`. Running header and layout whitespace stripped, empty pages skipped |
| 2. Split | `ingest.split` | 84 page Documents | 228 chunk Documents, 1200 chars with 200 overlap, page metadata copied onto each |
| 3. Index | `index.build_index` | 228 chunks | vector store: each chunk's text, metadata and embedding held together in memory |

## Once per question

| Step | Code | In | Out |
|---|---|---|---|
| 4. Retrieve | `index.search` | question, store | `hits`: top 5 chunks, each with a similarity score. The question is embedded and compared against all 228 |
| 5. Prompt | `qa.format_context` | hits | passages labelled `[page 17] ...`, plus the rules and the question |
| 6. Ask | `qa.answer` | prompt | `ModelAnswer`: the model's `found`, `answer`, `citations`. **Untrusted** |
| 7. Verify | `qa._verify` | model citations, hits | citations only. A page never retrieved is dropped as fabricated; a snippet that is not an exact quote keeps its page and loses the quote |
| 8. Decide | `qa.answer` | model answer, verified citations | `found` recomputed: the model said found **and** the answer is non-empty **and** at least one citation survived |
| 9. Return | `qa.answer` | the above | `Result`: question, found, answer or `Data-Not-Found`, verified citations. The API serializes this same object |

## The trust boundary

`ModelAnswer` is what the model claims. `Result` is what the system vouches for. Everything between them is checking:

- `question` is ours; the model never sets it.
- `citations` pass through verification.
- `found` is recomputed, never copied. The model can only propose true; the code can still turn it false.
- `answer` is passed through when found, replaced by `Data-Not-Found` when not.

Keeping the two classes apart is deliberate: it puts the line where untrusted model output becomes a checked result in one visible place.

## Worked example

Model returns:

```
found: true
answer: "Hosting runs on GCP."
citations: [ page 17, "production infrastructure is hosted on GCP"
             page 40, "we also use AWS" ]
```

Verification: page 17 was retrieved and the sentence is really in that chunk, so it stays. Page 40 was never retrieved, so it is dropped. One citation survives, the answer has text, the model said found, so `found` is true.

Had both citations been dropped, `found` would flip to false and the answer would become `Data-Not-Found`, regardless of the model claiming success. An answer nothing backs cannot be checked, and grounding is the point.
