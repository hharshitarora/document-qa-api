"""Citation verification and the found rule: the checks that make grounding real."""

import pytest
from langchain_core.documents import Document

from app.qa import NOT_FOUND, Citation, ModelAnswer, _verify, answer, answer_all, format_context


class TestFormatContext:
    def test_labels_each_passage_with_its_locator(self, hits):
        context = format_context(hits)
        assert context.startswith("[page 17]")
        assert "hosted on GCP" in context


class TestVerify:
    def test_exact_quote_survives(self, hits):
        citations = [Citation(location="page 17", snippet="Production infrastructure is hosted on GCP.")]

        verified = _verify(citations, hits)

        assert len(verified) == 1
        assert verified[0].snippet == "Production infrastructure is hosted on GCP."

    def test_location_never_retrieved_is_dropped(self, hits):
        """A page the model was not shown cannot have been read."""
        citations = [Citation(location="page 40", snippet="we also use AWS")]

        assert _verify(citations, hits) == []

    def test_paraphrased_quote_keeps_the_page_and_loses_the_snippet(self, hits):
        citations = [Citation(location="page 17", snippet="GCP hosts everything we run")]

        verified = _verify(citations, hits)

        assert len(verified) == 1
        assert verified[0].location == "page 17"
        assert verified[0].snippet == ""

    def test_quote_matching_is_insensitive_to_spacing_and_case(self, hits):
        citations = [Citation(location="page 17", snippet="PRODUCTION   infrastructure is hosted   on gcp.")]

        assert _verify(citations, hits)[0].snippet != ""

    def test_quote_spanning_two_chunks_of_one_page_verifies(self):
        """A page is often several chunks, and a quote from any of them is legitimate."""
        page_hits = [
            (Document(page_content="First half of the control.", metadata={"locator": "page 3"}), 0.4),
            (Document(page_content="Second half names the provider.", metadata={"locator": "page 3"}), 0.4),
        ]
        citations = [Citation(location="page 3", snippet="Second half names the provider.")]

        assert _verify(citations, page_hits)[0].snippet != ""

    def test_empty_snippet_keeps_the_location(self, hits):
        verified = _verify([Citation(location="page 17", snippet="  ")], hits)
        assert verified[0].location == "page 17"
        assert verified[0].snippet == ""

    def test_whitespace_around_the_location_is_tolerated(self, hits):
        citations = [Citation(location=" page 17 ", snippet="Backups run nightly.")]
        assert _verify(citations, hits)[0].location == "page 17"


def fake_model(monkeypatch, parsed: ModelAnswer | None, parsing_error: str | None = None):
    """Replace the chat model so no network call happens.

    `with_structured_output(..., include_raw=True)` returns a dict of parsed, raw and
    parsing_error, so the fake has to imitate that shape rather than return text.
    """

    class FakeRaw:
        usage_metadata = {"input_tokens": 100, "output_tokens": 20}

    class FakeStructured:
        async def ainvoke(self, _prompt):
            return {"parsed": parsed, "raw": FakeRaw(), "parsing_error": parsing_error}

    class FakeChat:
        def __init__(self, **_kwargs):
            pass

        def with_structured_output(self, _schema, include_raw=False):
            return FakeStructured()

    monkeypatch.setattr("app.qa.ChatOpenAI", FakeChat)


@pytest.mark.asyncio
class TestAnswer:
    async def test_answer_with_a_verified_citation(self, monkeypatch, hits):
        fake_model(
            monkeypatch,
            ModelAnswer(
                found=True,
                answer="Hosting runs on GCP.",
                citations=[Citation(location="page 17", snippet="Production infrastructure is hosted on GCP.")],
            ),
        )

        result, usage = await answer("Which cloud providers?", hits)

        assert result.found is True
        assert result.answer == "Hosting runs on GCP."
        assert result.citations[0].location == "page 17"
        assert result.error is None
        assert usage["input_tokens"] == 100

    async def test_claimed_answer_with_only_fabricated_citations_becomes_not_found(self, monkeypatch, hits):
        """An answer nothing backs cannot be checked, and grounding is the point."""
        fake_model(
            monkeypatch,
            ModelAnswer(
                found=True,
                answer="We use AWS.",
                citations=[Citation(location="page 99", snippet="invented")],
            ),
        )

        result, _usage = await answer("Which cloud providers?", hits)

        assert result.found is False
        assert result.answer == NOT_FOUND
        assert result.citations == []

    async def test_claimed_answer_with_empty_text_becomes_not_found(self, monkeypatch, hits):
        fake_model(
            monkeypatch,
            ModelAnswer(found=True, answer="   ", citations=[Citation(location="page 17", snippet="Backups run nightly.")]),
        )

        result, _usage = await answer("Anything?", hits)

        assert result.found is False
        assert result.answer == NOT_FOUND

    async def test_model_refusal_is_reported_as_not_found(self, monkeypatch, hits):
        fake_model(monkeypatch, ModelAnswer(found=False, answer="", citations=[]))

        result, _usage = await answer("What is the CEO's home address?", hits)

        assert result.found is False
        assert result.answer == NOT_FOUND
        assert result.error is None, "a refusal is an answer, not a failure"

    async def test_unparseable_response_raises(self, monkeypatch, hits):
        """A response that does not fit the contract is a fault, not an absence of data."""
        fake_model(monkeypatch, None, parsing_error="schema mismatch")

        with pytest.raises(ValueError, match="unparseable"):
            await answer("Which cloud providers?", hits)


@pytest.mark.asyncio
class TestAnswerAll:
    async def test_answers_keep_the_order_asked(self, monkeypatch, hits):
        fake_model(
            monkeypatch,
            ModelAnswer(
                found=True,
                answer="Hosting runs on GCP.",
                citations=[Citation(location="page 17", snippet="Backups run nightly.")],
            ),
        )
        monkeypatch.setattr("app.qa.search", lambda _store, _question: hits)

        questions = [f"question {number}?" for number in range(6)]
        results, usage = await answer_all(questions, store=object())

        assert [result.question for result in results] == questions
        assert usage["input_tokens"] == 600, "token usage is summed across the batch"

    async def test_one_failure_does_not_sink_the_batch(self, monkeypatch, hits):
        """A questionnaire of 50 should not be lost to one bad call."""
        fake_model(
            monkeypatch,
            ModelAnswer(
                found=True,
                answer="Hosting runs on GCP.",
                citations=[Citation(location="page 17", snippet="Backups run nightly.")],
            ),
        )

        def flaky_search(_store, question):
            if question == "boom?":
                raise RuntimeError("embedding call failed")
            return hits

        monkeypatch.setattr("app.qa.search", flaky_search)

        results, _usage = await answer_all(["fine?", "boom?", "also fine?"], store=object())

        assert [result.found for result in results] == [True, False, True]
        assert results[1].error is not None
        assert "embedding call failed" in results[1].error
        assert results[1].answer == NOT_FOUND
