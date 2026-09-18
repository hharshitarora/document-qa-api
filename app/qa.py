"""Turn retrieved passages into a checked, structured answer.

The prompt and the answer contract are written by hand rather than taken from a
prebuilt chain: grounding and the refusal rule decide whether an answer can be
trusted, so they are worth owning. See decisions.md and flow.md.
"""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings

# Their own sample answer file uses this exact string where the document has no answer.
NOT_FOUND = "Data-Not-Found"


class Citation(BaseModel):
    page: int = Field(description="page number of the passage used")
    snippet: str = Field(description="exact sentence copied from that passage")


class ModelAnswer(BaseModel):
    """The shape the model must fill in. Untrusted: every field is a claim."""

    found: bool = Field(description="true only if the passages contain the answer")
    answer: str = Field(description="the answer, or empty when found is false")
    citations: list[Citation] = Field(default_factory=list)


class Result(BaseModel):
    """What the system vouches for. The API serializes this same object."""

    question: str
    found: bool
    answer: str
    citations: list[Citation]


PROMPT = """You answer questions about a single document.

Rules:
- Use only the passages below. No outside knowledge, no guessing.
- Set found to false when the passages do not answer the question, and leave answer empty.
- Every citation must copy an exact sentence from a passage, with that passage's page number.

Passages:
{context}

Question: {question}
"""


def format_context(hits: list[tuple[Document, float]]) -> str:
    """One labelled block per passage, so the model can attribute what it used."""
    return "\n\n".join(
        f"[page {doc.metadata['page']}] {doc.page_content}" for doc, _score in hits
    )


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def _verify(citations: list[Citation], hits: list[tuple[Document, float]]) -> list[Citation]:
    """Check citations against the passages actually retrieved.

    A page that was never retrieved cannot have been read, so that citation is
    dropped. A snippet that is not a substring of its page's chunk was reworded, so
    the page is kept and the quote dropped rather than shipping a quote the document
    does not contain. Whitespace and case are normalised first, or every quote would
    fail on a stray double space.
    """
    retrieved = {doc.metadata["page"]: _normalise(doc.page_content) for doc, _score in hits}
    verified: list[Citation] = []

    for citation in citations:
        source = retrieved.get(citation.page)
        if source is None:
            continue
        snippet = citation.snippet.strip()
        exact = bool(snippet) and _normalise(snippet) in source
        verified.append(Citation(page=citation.page, snippet=snippet if exact else ""))

    return verified


def answer(question: str, hits: list[tuple[Document, float]]) -> tuple[Result, dict]:
    """Answer from the retrieved passages. Returns the result and token usage."""
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        # Same question and same passages should give the same answer: this is lookup,
        # not creative writing.
        temperature=0,
        # include_raw keeps the underlying response, and with it the token counts that
        # structured output would otherwise discard.
    ).with_structured_output(ModelAnswer, include_raw=True)

    response = llm.invoke(PROMPT.format(context=format_context(hits), question=question))

    parsed: ModelAnswer | None = response["parsed"]
    if parsed is None:
        # A response that does not fit the contract is a fault, not an absence of
        # information, so it is raised rather than reported as not found.
        raise ValueError(f"model returned an unparseable response: {response['parsing_error']}")

    citations = _verify(parsed.citations, hits)

    # An answer nothing backs cannot be checked, and grounding is the point, so a
    # claimed answer with no surviving citation is downgraded to not found.
    found = parsed.found and bool(parsed.answer.strip()) and bool(citations)

    return (
        Result(
            question=question,
            found=found,
            answer=parsed.answer.strip() if found else NOT_FOUND,
            citations=citations if found else [],
        ),
        response["raw"].usage_metadata or {},
    )
