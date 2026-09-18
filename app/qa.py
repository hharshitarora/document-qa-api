"""Turn retrieved passages into an answer.

The prompt is written by hand rather than taken from a prebuilt chain: grounding and
the refusal rule are the parts that decide whether an answer can be trusted, so they
are worth owning. See decisions.md.
"""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import settings

# Their own sample answer file uses this exact string where the document has no answer.
NOT_FOUND = "Data-Not-Found"

PROMPT = """You answer questions about a single document.

Use only the passages below. Do not use outside knowledge, and do not guess.
If the passages do not contain the answer, say exactly: {not_found}

Passages:
{context}

Question: {question}
"""


def format_context(hits: list[tuple[Document, float]]) -> str:
    """One labelled block per passage, so the model can see where each came from."""
    return "\n\n".join(
        f"[page {doc.metadata['page']}] {doc.page_content}" for doc, _score in hits
    )


def answer(question: str, hits: list[tuple[Document, float]]) -> tuple[str, dict]:
    """Answer from the retrieved passages. Returns the text and the token usage."""
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        # Same question and same passages should give the same answer: this is lookup,
        # not creative writing.
        temperature=0,
    )
    response = llm.invoke(
        PROMPT.format(
            not_found=NOT_FOUND,
            context=format_context(hits),
            question=question,
        )
    )
    return response.text.strip(), response.usage_metadata or {}
