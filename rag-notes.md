# RAG notes

Plain-language notes on the pieces, added to as they come up.

**RAG.** Retrieval augmented generation. The model brings language and reasoning, the document brings facts. Retrieve the relevant passages first, then let the model answer from them only.

**Chunk.** A slice of the document's text. It is the unit of three things at once: what gets embedded, what gets retrieved, and what the model reads. Nothing more clever than that.

**Chunk size.** Counted in characters by the splitter, not tokens. Roughly four characters per token in English, so 1200 characters is about 300 tokens. A chunk should be just large enough to hold one complete idea, which in a SOC 2 report is one control description.

**Overlap.** The tail of one chunk is repeated at the head of the next. With size 1200 and overlap 200, chunk two starts at character 1000, not 1200. It is insurance against a split landing mid-sentence: "Customers are notified within" in one chunk and "24 hours of a confirmed breach" in the next answers nobody's question, and neither slice matches the question well. Cost is embedding some text twice, which is cheap.

**Embedding.** Text sent to an embedding model comes back as a vector, a list of numbers standing for its meaning. Similar meanings land near each other. Questions get embedded the same way, which is how a question finds its passages.

**Vector store.** Where those vectors live, with the text and metadata beside them. Its job is to answer "which chunks are closest to this question".

**k.** How many chunks come back per question. Too few and the answer sits just below the cut. Too many and the model has to find the answer inside pages of nearly relevant text, which measurably lowers accuracy. Typical range is 3 to 8.

**Metadata and citations.** The PDF loader puts the page number on every page Document, and the splitter copies that metadata onto every chunk it produces. That is the entire citation mechanism: the page travels with the text all the way to the answer.

**Choosing these numbers.** Defaults are a starting point, not a decision. The real basis is measurement: run the actual questions, look at what was retrieved, and check whether the passage holding the answer came back at all. Retrieved but wrong is a prompt problem. Not retrieved is a chunking or k problem.
