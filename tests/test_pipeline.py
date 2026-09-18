"""Document preparation and its cache."""

import json

from app import pipeline


def reset_cache():
    pipeline._cache.clear()


def counting_pipeline(monkeypatch) -> dict:
    """Count parses and index builds so the cache can be observed, not assumed."""
    calls = {"parsed": 0, "indexed": 0}

    real_load = pipeline.load_bytes

    def counted_load(data, filename):
        calls["parsed"] += 1
        return real_load(data, filename)

    def counted_build(chunks):
        calls["indexed"] += 1
        return object()

    monkeypatch.setattr(pipeline, "load_bytes", counted_load)
    monkeypatch.setattr(pipeline, "build_index", counted_build)
    return calls


DOCUMENT = json.dumps([{"answer": "US Central, on GCP."}]).encode()


class TestPrepareDocument:
    def test_first_call_parses_and_indexes(self, monkeypatch):
        reset_cache()
        calls = counting_pipeline(monkeypatch)

        _store, chunks, cached = pipeline.prepare_document(DOCUMENT, "kb.json")

        assert (calls["parsed"], calls["indexed"]) == (1, 1)
        assert chunks == 1
        assert cached is False

    def test_same_bytes_skip_parsing_as_well_as_embedding(self, monkeypatch):
        """Caching after parsing saved nothing measurable: the parse is the slow half."""
        reset_cache()
        calls = counting_pipeline(monkeypatch)

        pipeline.prepare_document(DOCUMENT, "kb.json")
        _store, chunks, cached = pipeline.prepare_document(DOCUMENT, "kb.json")

        assert (calls["parsed"], calls["indexed"]) == (1, 1), "nothing should run twice"
        assert cached is True
        assert chunks == 1, "the chunk count is remembered, not recomputed"

    def test_same_document_under_a_different_name_is_a_hit(self, monkeypatch):
        reset_cache()
        calls = counting_pipeline(monkeypatch)

        pipeline.prepare_document(DOCUMENT, "kb.json")
        _store, _chunks, cached = pipeline.prepare_document(DOCUMENT, "renamed-copy.json")

        assert cached is True, "the key is the bytes, not the filename"
        assert calls["parsed"] == 1

    def test_different_documents_are_separate_entries(self, monkeypatch):
        reset_cache()
        calls = counting_pipeline(monkeypatch)
        other = json.dumps([{"answer": "Frankfurt, on AWS."}]).encode()

        pipeline.prepare_document(DOCUMENT, "kb.json")
        _store, _chunks, cached = pipeline.prepare_document(other, "kb.json")

        assert cached is False, "a different document sharing a name is not a hit"
        assert calls["parsed"] == 2
        assert pipeline.cache_size() == 2

    def test_cache_is_bounded_and_evicts_the_oldest(self, monkeypatch):
        reset_cache()
        counting_pipeline(monkeypatch)
        monkeypatch.setattr(pipeline.settings, "cache_documents", 2)

        for number in range(3):
            body = json.dumps([{"answer": f"document {number}"}]).encode()
            pipeline.prepare_document(body, f"doc{number}.json")

        assert pipeline.cache_size() == 2

        first_again = json.dumps([{"answer": "document 0"}]).encode()
        _store, _chunks, cached = pipeline.prepare_document(first_again, "doc0.json")
        assert cached is False, "the oldest entry was evicted"
