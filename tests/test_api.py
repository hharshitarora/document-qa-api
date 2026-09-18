"""The endpoint, with the model and embeddings replaced.

Validation is tested for real; the happy path fakes document preparation and answering,
which is what the brief means by an integration test with mocked dependencies.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api import app
from app.qa import Citation, Result

client = TestClient(app)


def questions_file(questions: list) -> dict:
    return {"questions": ("questions.json", json.dumps(questions).encode(), "application/json")}


def document_file(name: str = "kb.json", payload: object = None) -> dict:
    body = json.dumps(payload if payload is not None else [{"answer": "US Central, on GCP."}]).encode()
    return {"document": (name, body, "application/json")}


class TestHealth:
    def test_reports_models_and_limits_without_the_key(self):
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["chat_model"]
        assert body["limits"]["max_questions"] > 0
        assert "api_key" not in json.dumps(body).lower().replace("openai_api_key", "")


class TestValidation:
    def test_unsupported_document_type(self):
        response = client.post(
            "/qa",
            files={"document": ("notes.txt", b"plain text", "text/plain"), **questions_file(["Where?"])},
        )

        assert response.status_code == 400
        assert "unsupported file type" in response.json()["detail"]

    def test_document_that_is_not_valid_json(self):
        response = client.post(
            "/qa",
            files={"document": ("kb.json", b'{"unterminated', "application/json"), **questions_file(["Where?"])},
        )

        assert response.status_code == 400
        assert "not valid JSON" in response.json()["detail"]

    def test_questions_file_that_is_not_valid_json(self):
        response = client.post(
            "/qa",
            files={**document_file(), "questions": ("questions.json", b"[[[", "application/json")},
        )

        assert response.status_code == 400
        assert "questions file is not valid JSON" in response.json()["detail"]

    def test_questions_object_without_a_questions_key(self):
        response = client.post("/qa", files={**document_file(), **questions_file({"nope": 1})})

        assert response.status_code == 400
        assert '"questions" key' in response.json()["detail"]

    def test_questions_file_holding_a_bare_string(self):
        response = client.post("/qa", files={**document_file(), **questions_file("Where?")})

        assert response.status_code == 400
        assert "must hold a list" in response.json()["detail"]

    def test_questions_object_with_a_questions_list(self, monkeypatch):
        """The brief only says "a file containing a list of questions", so both shapes work."""
        fake_pipeline(monkeypatch)

        response = client.post("/qa", files={**document_file(), **questions_file({"questions": ["Where?"]})})

        assert response.status_code == 200
        assert len(response.json()["results"]) == 1

    def test_question_that_is_not_text(self):
        response = client.post("/qa", files={**document_file(), **questions_file(["fine?", 42])})

        assert response.status_code == 400
        assert "question 2" in response.json()["detail"]

    def test_empty_questions_list(self):
        response = client.post("/qa", files={**document_file(), **questions_file([])})

        assert response.status_code == 400
        assert "no questions" in response.json()["detail"]

    def test_too_many_questions(self):
        response = client.post("/qa", files={**document_file(), **questions_file(["q?"] * 51)})

        assert response.status_code == 400
        assert "exceeds the limit" in response.json()["detail"]

    def test_neither_questions_file_nor_question_field(self):
        response = client.post("/qa", files=document_file())

        assert response.status_code == 400
        assert "question" in response.json()["detail"]

    def test_empty_document(self):
        response = client.post(
            "/qa",
            files={"document": ("kb.json", b"", "application/json"), **questions_file(["Where?"])},
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_document_over_the_size_limit(self, monkeypatch):
        monkeypatch.setattr("app.api.settings.max_upload_bytes", 1024)
        oversized = json.dumps([{"answer": "x" * 4000}]).encode()

        response = client.post(
            "/qa",
            files={"document": ("kb.json", oversized, "application/json"), **questions_file(["Where?"])},
        )

        assert response.status_code == 413
        assert "limit" in response.json()["detail"]


def fake_pipeline(monkeypatch, results: list[Result] | None = None):
    """Replace document preparation and answering: no embeddings, no model."""
    monkeypatch.setattr("app.api.prepare_document", lambda _data, _name: (object(), 7, False))

    async def fake_answer_all(questions, _store):
        prepared = results or [
            Result(
                question=question,
                found=True,
                answer="US Central, on GCP.",
                citations=[Citation(location="record 1", snippet="US Central, on GCP.")],
            )
            for question in questions
        ]
        return prepared, {"input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr("app.api.answer_all", fake_answer_all)


class TestQA:
    def test_answers_every_question_in_the_file(self, monkeypatch):
        fake_pipeline(monkeypatch)

        response = client.post(
            "/qa", files={**document_file(), **questions_file(["Where are the data centres?", "Backups?"])}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["document"] == "kb.json"
        assert body["chunks"] == 7
        assert body["cached"] is False
        assert len(body["results"]) == 2
        assert body["results"][0]["citations"][0]["location"] == "record 1"

    def test_single_question_field_instead_of_a_file(self, monkeypatch):
        fake_pipeline(monkeypatch)

        response = client.post("/qa", files=document_file(), data={"question": " Where are the data centres? "})

        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 1
        assert results[0]["question"] == "Where are the data centres?", "whitespace is trimmed"

    def test_a_failed_question_returns_200_with_an_error_on_that_result(self, monkeypatch):
        fake_pipeline(
            monkeypatch,
            results=[
                Result(question="fine?", found=True, answer="yes", citations=[]),
                Result(question="boom?", found=False, answer="Data-Not-Found", citations=[], error="model call failed"),
            ],
        )

        response = client.post("/qa", files={**document_file(), **questions_file(["fine?", "boom?"])})

        assert response.status_code == 200, "one bad question does not fail the batch"
        results = response.json()["results"]
        assert results[1]["error"] == "model call failed"

    def test_upstream_failure_is_a_502(self, monkeypatch):
        from app.errors import UpstreamError

        def broken(_data, _name):
            raise UpstreamError("embedding call failed: connection reset")

        monkeypatch.setattr("app.api.prepare_document", broken)

        response = client.post("/qa", files={**document_file(), **questions_file(["Where?"])})

        assert response.status_code == 502
        assert "embedding call failed" in response.json()["detail"]

    def test_cached_flag_is_passed_through(self, monkeypatch):
        monkeypatch.setattr("app.api.prepare_document", lambda _data, _name: (object(), 19, True))

        async def fake_answer_all(questions, _store):
            return [Result(question=q, found=False, answer="Data-Not-Found", citations=[]) for q in questions], {}

        monkeypatch.setattr("app.api.answer_all", fake_answer_all)

        response = client.post("/qa", files={**document_file(), **questions_file(["Where?"])})

        assert response.json()["cached"] is True
