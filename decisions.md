# Decisions

Running log, written as each choice is made. Feeds the README.

| # | Decision | Why | Tradeoff |
|---|---|---|---|
| 1 | LangChain for document loading, splitting and the vector store interface; retrieval, prompting and the answer schema written by hand | The brief suggests LangChain, and its loaders and splitters are commodity work; grounding and the output contract are the parts worth owning | More code than an off-the-shelf chain, and two layers to understand instead of one |
| 2 | Vectors held in memory, behind LangChain's vector store interface | One 84-page document is roughly 300 chunks, so exact search is instant and needs no extra service; the interface means Chroma or pgvector is a one-line swap | Nothing survives a restart, and it does not scale past what fits in memory |
| 3 | Built index cached in memory, keyed by a hash of the uploaded file | Re-uploading the same document skips embedding entirely, saving time and API spend | Cache is lost on restart, and memory grows with distinct documents unless evicted |

## Notes
- The brief's link to LangChain's question-answering tutorial now redirects to their Deep Agents RAG page. Built the plain retrieval pipeline the brief describes, not the agent framework.
- Deliberately not persisting document text to disk: this is customer compliance data, and storing it would raise retention and deletion questions a take-home should not create.
