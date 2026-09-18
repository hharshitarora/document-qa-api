"""HTTP interface over the pipeline.

Uploads are parsed in memory, never written to disk: document text is customer data.
Every limit lives in config, and library failures are translated into typed errors at
the edge, so responses carry a message rather than a stack trace.
"""

import json

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.errors import InputError, TooLargeError, UpstreamError
from app.pipeline import prepare_document
from app.qa import Result, answer_all

app = FastAPI(
    title="Document Q&A",
    description="Answer questions about an uploaded PDF or JSON document, with citations.",
    version="1.0.0",
)


class QAResponse(BaseModel):
    document: str
    chunks: int
    # True when this document was already indexed, so no embedding calls were made.
    cached: bool
    results: list[Result]


@app.exception_handler(TooLargeError)
async def handle_too_large(_request: Request, exc: TooLargeError) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(InputError)
async def handle_input_error(_request: Request, exc: InputError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(UpstreamError)
async def handle_upstream_error(_request: Request, exc: UpstreamError) -> JSONResponse:
    # The caller did nothing wrong, so this is reported as a gateway failure rather
    # than a bad request, and the message says which call failed.
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def parse_questions(data: bytes) -> list[str]:
    """Questions from an uploaded JSON file.

    The brief says only "a file containing a list of questions", so both plausible
    shapes are accepted: a bare list, or an object with a "questions" key. Items may
    be strings or objects carrying a "question" field. Anything else is rejected
    rather than silently skipped.
    """
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputError(f"questions file is not valid JSON: {exc}") from exc

    if isinstance(payload, dict):
        if "questions" not in payload:
            raise InputError('questions file must hold a list, or an object with a "questions" key')
        payload = payload["questions"]
    if not isinstance(payload, list):
        raise InputError('questions file must hold a list, or an object with a "questions" list')

    questions: list[str] = []
    for position, item in enumerate(payload, start=1):
        text = item.get("question", "") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip():
            raise InputError(f"question {position} is empty or not text")
        questions.append(text.strip())

    return questions


def check_questions(questions: list[str]) -> list[str]:
    if not questions:
        raise InputError("no questions were provided")
    if len(questions) > settings.max_questions:
        raise InputError(
            f"{len(questions)} questions exceeds the limit of {settings.max_questions} per request"
        )
    return questions


async def read_upload(upload: UploadFile, label: str) -> bytes:
    """Read an upload, refusing anything over the ceiling.

    The body is already in memory by the time this runs, so the ceiling bounds what is
    processed rather than what is received. Enforcing it during the read needs a
    streaming request body, noted as a limitation in the README.
    """
    data = await upload.read()
    if not data:
        raise InputError(f"{label} is empty")
    if len(data) > settings.max_upload_bytes:
        raise TooLargeError(
            f"{label} is {len(data) // 1024 // 1024}MB, over the "
            f"{settings.max_upload_bytes // 1024 // 1024}MB limit"
        )
    return data


@app.get("/health")
def health() -> dict:
    """Liveness check, and the models and limits in use. No key material is exposed."""
    return {
        "status": "ok",
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "limits": {
            "max_upload_mb": settings.max_upload_bytes // 1024 // 1024,
            "max_questions": settings.max_questions,
            "max_concurrent_questions": settings.max_concurrent_questions,
        },
    }


@app.post("/qa", response_model=QAResponse)
async def qa(
    document: UploadFile = File(..., description="the document to answer from, .pdf or .json"),
    questions: UploadFile | None = File(None, description="JSON file holding a list of questions"),
    question: str | None = Form(None, description="a single question, instead of a file"),
) -> QAResponse:
    """Answer questions about the uploaded document.

    Questions arrive as a file, which is what the brief specifies and what a
    questionnaire looks like, or as a single `question` field for one-off use.
    """
    if questions is None and not (question or "").strip():
        raise InputError('send a questions file, or a "question" field')

    question_list = (
        parse_questions(await read_upload(questions, "questions file"))
        if questions is not None
        else [question.strip()]
    )
    check_questions(question_list)

    data = await read_upload(document, "document")
    store, chunk_count, cached = prepare_document(data, document.filename or "")

    results, _usage = await answer_all(question_list, store)
    return QAResponse(
        document=document.filename or "",
        chunks=chunk_count,
        cached=cached,
        results=results,
    )
