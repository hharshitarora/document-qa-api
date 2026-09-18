"""Document to searchable index, with caching.

One place owns the order of work, so the API stays a thin HTTP layer. The cache is
checked before parsing, not after: a repeat upload should skip reading the PDF as well
as embedding it, and on the sample report parsing is the slower half.
"""

import hashlib
from collections import OrderedDict

from langchain_core.vectorstores import InMemoryVectorStore

from app.config import settings
from app.index import build_index
from app.ingest import load_bytes, split

# Prepared documents, newest last, keyed by a hash of the bytes rather than the
# filename: the same report uploaded under two names is the same document, and two
# different documents can share a name. Vectors only, only in memory, nothing on disk.
_cache: OrderedDict[str, tuple[InMemoryVectorStore, int]] = OrderedDict()


def prepare_document(data: bytes, filename: str) -> tuple[InMemoryVectorStore, int, bool]:
    """Return a searchable index for this document, its chunk count, and a cache flag."""
    key = hashlib.sha256(data).hexdigest()

    cached = _cache.get(key)
    if cached is not None:
        _cache.move_to_end(key)
        store, chunk_count = cached
        return store, chunk_count, True

    chunks = split(load_bytes(data, filename))
    store = build_index(chunks)

    _cache[key] = (store, len(chunks))
    while len(_cache) > settings.cache_documents:
        _cache.popitem(last=False)  # evict the least recently used

    return store, len(chunks), False


def cache_size() -> int:
    return len(_cache)
