import faiss
import hashlib
import logging
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path
from sentence_transformers import SentenceTransformer
from threading import Lock

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    RAG_DB_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    VECTOR_CANDIDATES,
    FTS_CANDIDATES,
)
from rag.fts import search_fts
from rag.fusion import reciprocal_rank_fusion
from rag.query_expansion import expand_query
from rag.store import load_chunks, load_generation_metadata

logger = logging.getLogger(__name__)

model = None
_model_lock = Lock()

# Cached chunk generation and its validated vector identity.
index = None
chunks = []
generation_metadata = None
validated_faiss_sha256 = None
# Serialize readiness and retain the lock through fallback on a failed repair.
_retrieval_lock = Lock()


def _get_model():
    """Construct the query model once, allowing retries after a failed attempt."""
    global model
    if model is None:
        with _model_lock:
            if model is None:
                model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def _ensure_chunks_loaded(*, reload=False):
    """Load chunks and identity independently, preserving lexical fallback."""
    global chunks, index, generation_metadata, validated_faiss_sha256
    if chunks and generation_metadata is not None and not reload:
        return True
    index = None
    generation_metadata = None
    validated_faiss_sha256 = None
    try:
        chunks = load_chunks(RAG_DB_PATH)
    except Exception as exc:
        # Never resolve new IDs through an older cached mapping.
        chunks = []
        logger.warning("Chunk loading failed: %s", exc)
        return False
    try:
        generation_metadata = load_generation_metadata(RAG_DB_PATH)
        if generation_metadata is None:
            logger.warning("Generation metadata missing; vector compatibility is unproven")
    except Exception as exc:
        logger.warning("Generation metadata loading failed: %s", exc)
    return True


def _sha256_file(path: Path) -> str:
    """Hash the published FAISS bytes with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generation_is_validated():
    """Check cached readiness without touching storage or the query model."""
    return (
        index is not None
        and generation_metadata is not None
        and validated_faiss_sha256 == generation_metadata["faiss_sha256"]
        and len(chunks) == generation_metadata["chunk_count"] == index.ntotal
    )


def _load_validated_index(index_path):
    """Publish a candidate in memory only after proving generation identity."""
    global index, validated_faiss_sha256
    if generation_metadata is None:
        raise ValueError("Vector generation incompatible: missing valid generation metadata")
    expected_count = generation_metadata["chunk_count"]
    if len(chunks) != expected_count:
        raise ValueError("Vector generation incompatible: loaded chunk count mismatch")
    candidate_index = faiss.read_index(str(index_path))
    if candidate_index.ntotal != expected_count:
        raise ValueError("Vector generation incompatible: FAISS count mismatch")
    faiss_sha256 = _sha256_file(index_path)
    if faiss_sha256 != generation_metadata["faiss_sha256"]:
        raise ValueError("Vector generation incompatible: FAISS SHA-256 mismatch")
    index = candidate_index
    validated_faiss_sha256 = faiss_sha256


def _ensure_index_exists():
    """Prepare shared artifacts before searches; preserve chunks on vector failure."""
    global index, validated_faiss_sha256
    if _generation_is_validated():
        return True

    index_path = Path(__file__).parent.parent / FAISS_INDEX_PATH
    index = None
    validated_faiss_sha256 = None
    chunks_ready = _ensure_chunks_loaded()
    if chunks_ready:
        try:
            _load_validated_index(index_path)
            return True
        except Exception as exc:
            logger.warning("Vector index loading failed; attempting rebuild: %s", exc)

    try:
        from rag.build_index import build_index
        build_index()
    except Exception as exc:
        logger.warning("Vector index rebuild failed: %s", exc)
        return False

    # Rebuilding can change every position: reload chunks even if FAISS fails.
    if not _ensure_chunks_loaded(reload=True):
        return False
    try:
        _load_validated_index(index_path)
        return True
    except Exception as exc:
        logger.warning("Vector index loading after rebuild failed: %s", exc)
        return False


def _valid_candidate_ids(candidates) -> list[int]:
    """Keep unique positions in the loaded chunks, in first-occurrence order."""
    valid = []
    seen = set()
    for position in candidates:
        if isinstance(position, bool) or not isinstance(position, Integral):
            continue
        if 0 <= position < len(chunks) and position not in seen:
            seen.add(position)
            valid.append(int(position))
    return valid


def _vector_search_ids(query: str, limit: int) -> list[int]:
    """Search the original query with the existing normalized FAISS algorithm."""
    if limit <= 0:
        return []
    # Artifact readiness belongs to the caller, never to a parallel search.
    if index is None or len(chunks) == 0:
        return []
    
    candidate_count = min(limit, index.ntotal, len(chunks))
    if candidate_count <= 0:
        return []

    q_emb = _get_model().encode([query])
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, candidate_count)
    return _valid_candidate_ids(ids[0][:candidate_count])


def retrieve_vector(query: str, limit=None) -> list[dict]:
    """Retrieve vector-only contexts, including for frozen benchmark reproduction."""
    limit = TOP_K if limit is None else limit
    with _retrieval_lock:
        if limit > 0 and not _ensure_index_exists():
            return []
    ids = _vector_search_ids(query, limit)
    return [chunks[position] for position in ids]


def retrieve(query: str) -> list[dict]:
    """Expand lexical inputs, search concurrently, and fuse ranked chunk positions."""
    try:
        lexical_inputs = expand_query(query)
    except Exception as exc:
        logger.warning("Query expansion failed; using the original query: %s", exc)
        lexical_inputs = [query]

    with ExitStack() as readiness:
        readiness.enter_context(_retrieval_lock)
        vector_ready = _ensure_index_exists()
        if not chunks:
            return []
        if vector_ready:
            # Loaded artifacts are reused by subsequent healthy requests.
            readiness.close()
        # Otherwise, prevent another repair from replacing FTS/chunks until
        # this fallback request finishes searching and mapping its results.

        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(_vector_search_ids, query, VECTOR_CANDIDATES)
            fts_future = executor.submit(search_fts, lexical_inputs, limit=FTS_CANDIDATES)
            rankings = []
            for name, future in (("Vector", vector_future), ("FTS", fts_future)):
                try:
                    rankings.append(_valid_candidate_ids(future.result()))
                except Exception as exc:
                    logger.warning("%s retrieval failed: %s", name, exc)
                    rankings.append([])

        fused_ids = reciprocal_rank_fusion(rankings, limit=TOP_K)
        return [chunks[position] for position in fused_ids]


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt):
    """Query Ollama LLM."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
