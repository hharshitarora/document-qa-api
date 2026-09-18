"""HTTP interface over the pipeline.

Uploads are parsed in memory, never written to disk: document text is customer data.
Validation, limits and concurrency are deliberately absent here and handled next.
"""

import json

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.index import build_index, search
from app.ingest import load_bytes, split
from app.qa import Result, answer

app = FastAPI(
    title="Document Q&A",
    description="Answer questions about an uploaded PDF or JSON document, with citations.",
    version="0.1.0",
)


class QAResponse(BaseModel):
    document: str
    chunks: int
    results: list[Result]


def parse_questions(data: bytes) -> list[str]:
    """Questions from an uploaded JSON file.

    The brief says only "a file containing a list of questions", so both plausible
    shapes are accepted: a bare list, or an object with a "questions" key. Items may
    be strings or objects carrying a "question" field.
    """
    payload = json.loads(data)
    if isinstance(payload, dict):
        payload = payload.get("questions", [])

    questions: list[str] = []
    for item in payload if isinstance(payload, list) else []:
        text = item.get("question", "") if isinstance(item, dict) else str(item)
        if text.strip():
            questions.append(text.strip())
    return questions


@app.get("/health")
def health() -> dict:
    """Liveness check, and the models in use. No key material is exposed."""
    return {
        "status": "ok",
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/qa", response_model=QAResponse)
async def qa(
    document: UploadFile = File(..., description="the document to answer from, .pdf or .json"),
    questions: UploadFile = File(..., description="JSON file holding a list of questions"),
) -> QAResponse:
    """Answer every question in the questions file from the uploaded document."""
    question_list = parse_questions(await questions.read())
    chunks = split(load_bytes(await document.read(), document.filename or ""))
    store = build_index(chunks)

    results: list[Result] = [
        answer(question, search(store, question))[0] for question in question_list
    ]

    return QAResponse(document=document.filename or "", chunks=len(chunks), results=results)
