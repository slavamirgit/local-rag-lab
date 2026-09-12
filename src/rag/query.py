import faiss
import logging
import pickle
import requests
import sys
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
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

logger = logging.getLogger(__name__)

model = SentenceTransformer(EMBEDDING_MODEL)

# Global variables for index and chunks
index = None
chunks = []


def _ensure_index_exists():
    """Ensure FAISS index exists, build it if it doesn't."""
    global index, chunks
    
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH
    
    # Check if index exists
    if index_path.exists() and chunks_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error loading existing index: {e}")
            print("Rebuilding index...")
    
    # Index doesn't exist or failed to load, build it
    print("📦 Index not found. Building index from documents...")
    try:
        from rag.build_index import build_index
        build_index()
        
        # Load the newly created index
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            print("✅ Index built and loaded successfully")
            return True
        else:
            print("❌ Failed to build index. No documents found or error occurred.")
            from config import DOCUMENTS_DIR
            docs_path = src_dir / DOCUMENTS_DIR
            print(f"   Check that documents exist in: {docs_path}")
            return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize index on module load
_ensure_index_exists()


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
    # Ensure index exists before retrieving
    if index is None or len(chunks) == 0:
        if not _ensure_index_exists():
            return []
    
    if index is None or len(chunks) == 0:
        return []
    
    candidate_count = min(limit, index.ntotal, len(chunks))
    if candidate_count <= 0:
        return []

    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, candidate_count)
    return _valid_candidate_ids(ids[0][:candidate_count])


def retrieve_vector(query: str, limit=None) -> list[dict]:
    """Retrieve vector-only contexts, including for frozen benchmark reproduction."""
    ids = _vector_search_ids(query, TOP_K if limit is None else limit)
    return [chunks[position] for position in ids]


def retrieve(query: str) -> list[dict]:
    """Expand lexical inputs, search concurrently, and fuse ranked chunk positions."""
    try:
        lexical_inputs = expand_query(query)
    except Exception as exc:
        logger.warning("Query expansion failed; using the original query: %s", exc)
        lexical_inputs = [query]

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
