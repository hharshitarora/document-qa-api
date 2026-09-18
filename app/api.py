"""HTTP interface over the pipeline.

Uploads are parsed in memory, never written to disk: document text is customer data.
Every limit lives in config, and library failures are translated into typed errors at
the edge, so responses carry a message rather than a stack trace.
"""

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.errors import InputError, TooLargeError, UpstreamError
from app.logs import estimate_cost, log_event, setup_logging, timed
from app.pipeline import prepare_document
from app.qa import Result, answer_all

setup_logging()

STATIC = Path(__file__).resolve().parent.parent / "static"

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request: identifier, route, status and duration.

    The id is echoed in the response header so a reported problem can be found in the
    logs. No paths carry document or question text, so nothing sensitive is logged.
    """
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    with timed() as elapsed:
        response = await call_next(request)

    log_event(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=elapsed["ms"],
    )
    response.headers["x-request-id"] = request_id
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The minimal upload page. The API explorer lives at /docs."""
    return FileResponse(STATIC / "index.html")


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
    request: Request,
    document: UploadFile = File(..., description="the document to answer from, .pdf or .json"),
    questions: UploadFile | None = File(None, description="JSON file holding a list of questions"),
    question: str | None = Form(None, description="a single question, instead of a file"),
) -> QAResponse:
    """Answer questions about the uploaded document.

    Questions arrive as a file, which is what the brief specifies and what a
    questionnaire looks like, or as a single `question` field for one-off use.
    """
    typed = (question or "").strip()
    if questions is None and not typed:
        raise InputError('send a questions file, or a "question" field')
    if questions is not None and typed:
        # Silently preferring one over the other is how a question typed after a file was
        # chosen gets ignored without explanation.
        raise InputError('send either a questions file or a "question" field, not both')

    question_list = (
        parse_questions(await read_upload(questions, "questions file"))
        if questions is not None
        else [typed]
    )
    check_questions(question_list)

    data = await read_upload(document, "document")

    with timed() as indexing:
        store, chunk_count, cached = prepare_document(data, document.filename or "")

    with timed() as answering:
        results, usage = await answer_all(question_list, store)

    log_event(
        "answered",
        request_id=getattr(request.state, "request_id", None),
        document=document.filename or "",
        document_bytes=len(data),
        chunks=chunk_count,
        cached=cached,
        questions=len(question_list),
        answered=sum(1 for result in results if result.found),
        failed=sum(1 for result in results if result.error),
        index_ms=indexing["ms"],
        answer_ms=answering["ms"],
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        estimated_cost_usd=estimate_cost(usage.get("input_tokens", 0), usage.get("output_tokens", 0)),
    )

    return QAResponse(
        document=document.filename or "",
        chunks=chunk_count,
        cached=cached,
        results=results,
    )
