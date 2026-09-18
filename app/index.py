"""Embed chunks and search them.

Vectors are held in memory behind LangChain's vector store interface: one document is
a few hundred chunks, so exact search is instant and needs no extra service, and
swapping in Chroma or pgvector later is a change in this file alone. See decisions.md.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.errors import UpstreamError

# How many chunks a question retrieves. Their questions often span several facts, so
# the answer can sit across passages; 5 is about 1500 tokens of context. See decisions.md.
TOP_K = 5


def build_index(chunks: list[Document]) -> InMemoryVectorStore:
    """Embed every chunk once and return a searchable store."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    store = InMemoryVectorStore(embeddings)
    try:
        store.add_documents(chunks)
    except Exception as exc:
        raise UpstreamError(f"embedding call failed: {exc}") from exc
    return store


def search(store: InMemoryVectorStore, question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    """Top-k chunks for a question, each with its similarity score.

    Scores come back alongside the text because a weak best match is itself a signal:
    it is what tells the answering step that the document may not cover the question.
    """
    try:
        return store.similarity_search_with_score(question, k=k)
    except Exception as exc:
        raise UpstreamError(f"embedding call failed while retrieving: {exc}") from exc
