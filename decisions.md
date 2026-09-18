# Decisions

Running log, written as each choice is made. Feeds the README.

| # | Decision | Why | Tradeoff |
|---|---|---|---|
| 1 | Retrieval augmented generation rather than prompting the model on its own | An LLM knows nothing about a customer's internal documents, so the document has to supply the facts and the model only supplies the reasoning and the wording | Answer quality is now capped by retrieval quality: if the right passage is not retrieved, a capable model still cannot answer |
| 2 | LangChain for document loading, splitting and the vector store interface; retrieval, prompting and the answer schema written by hand | The brief suggests LangChain, and its loaders and splitters are commodity work; grounding and the output contract are the parts worth owning | More code than an off-the-shelf chain, and two layers to understand instead of one |
| 3 | Vectors held in memory, behind LangChain's vector store interface | One 84-page document is roughly 300 chunks, so exact search is instant and needs no extra service; the interface means Chroma or pgvector is a one-line swap | Nothing survives a restart, and it does not scale past what fits in memory |
| 4 | Built index cached in memory, keyed by a hash of the uploaded file | Re-uploading the same document skips embedding entirely, saving time and API spend | Cache is lost on restart, and memory grows with distinct documents unless evicted |
| 5 | Chunks of 1200 characters | Their sample is a SOC 2 report, where a self-contained unit is one control description running a paragraph or two; 1200 characters is about 300 tokens, just above that, so a control stays intact | Larger chunks drag neighbouring controls into the prompt and blunt retrieval precision |
| 6 | Overlap of 200 characters | Roughly 15 percent, enough that a sentence straddling a split still appears complete in one of the two chunks | Some text is embedded twice, a small cost in tokens and storage |
| 7 | Retrieve 5 chunks per question | Their questions often span several facts, so the answer can sit across passages; 5 is about 1500 tokens of context, cheap against the budget | Fewer would risk missing a passage, more would bury the answer in nearly relevant text |
| 8 | Split page by page, never joining pages into one text | Keeps the page number exact on every chunk, so a citation points at "page 43" rather than at the document; precision of a citation is capped by the unit kept | A control running across a page boundary gets split with no overlap to rescue it, since overlap only applies within a page |
| 9 | Read the PDF with `pypdf` directly instead of a LangChain loader | The PDF loaders live in `langchain-community`, which is being sunset, and the maintained alternatives each cost something: PyMuPDF is AGPL, Docling downloads models, Unstructured needs system packages. Reading pages directly is a few lines and keeps control of citation metadata | One less framework convenience, and no table-aware parsing: tables flatten into text lines |
| 10 | Collapse runs of whitespace at load time | pypdf preserves layout spacing, so words arrive separated by two or three spaces; measured at 23% of all characters in the sample report, meaning chunks held a quarter less real content and retrieved chunks spent tokens on padding | Visual layout is lost, so table columns run together, though this parser flattens tables regardless |
| 11 | Detect and strip the running page header at load time | The sample report repeats the same title line on all 84 pages, which appeared in 84 of 232 chunks; identical text everywhere adds no information and pulls every chunk towards the same generic meaning | Detection is heuristic: a document whose real content happens to repeat on a third of its pages would lose it |
| 12 | Answer and not-found state carried as separate fields in a structured response, rather than a sentinel string inside free text | A prompt asking for `Data-Not-Found` "if the passages do not contain the answer" produced a correct answer with the sentinel appended to it, so a reader could not tell an answer from a refusal; separate fields make that contradiction unrepresentable instead of merely discouraged | A structured call is stricter and can fail to parse, so it needs its own error path |

Starting points, to be checked in Phase 4 against what actually gets retrieved for their 5 questions.

## Notes
- The brief's link to LangChain's question-answering tutorial now redirects to their Deep Agents RAG page. Built the plain retrieval pipeline the brief describes, not the agent framework.
- Deliberately not persisting document text to disk: this is customer compliance data, and storing it would raise retention and deletion questions a take-home should not create.
