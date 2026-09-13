import faiss
from contextlib import contextmanager
import logging
import os
import pickle
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from rag.fts import build_fts_index
from rag.store import build_rag_db
from config import FAISS_INDEX_PATH, CHUNKS_PATH, FTS_INDEX_PATH, RAG_DB_PATH

logger = logging.getLogger(__name__)


@contextmanager
def _staged_artifact(path):
    """Reserve a sibling file and clean up any unpublished data on exit."""
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as file:
        staged_path = Path(file.name)
    try:
        yield staged_path
    finally:
        # SQLite can leave sidecars behind when construction fails.
        for suffix in ("", "-journal", "-wal", "-shm"):
            temporary = Path(str(staged_path) + suffix)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # A cleanup error must not obscure the construction exception.
                logger.warning("Could not remove staging file %s", temporary, exc_info=True)


def build_index():
    """Build FAISS, legacy artifacts, and rag.db from the same ordered chunks."""
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH
    fts_path = Path(FTS_INDEX_PATH)
    rag_path = Path(RAG_DB_PATH)
    
    print("📥 Loading documents...")
    documents = ingest_documents()

    if not documents:
        print("❌ No documents found. Please add documents to the docs directory.")
        return

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)

    print("🧠 Generating embeddings...")
    embeddings = embed_chunks(chunks)

    print("📦 Creating FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print("💾 Saving...")
    # FTS has always created its parent directory and used cwd-relative paths.
    fts_path.parent.mkdir(parents=True, exist_ok=True)
    rag_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        _staged_artifact(index_path) as staged_index,
        _staged_artifact(chunks_path) as staged_chunks,
        _staged_artifact(fts_path) as staged_fts,
        _staged_artifact(rag_path) as staged_rag,
    ):
        faiss.write_index(index, str(staged_index))
        with staged_chunks.open("wb") as f:
            pickle.dump(chunks, f)

        print("📦 Creating FTS index...")
        build_fts_index(chunks, index_path=staged_fts)

        print("📦 Creating chunk database...")
        build_rag_db(chunks, staged_rag)

        # Publish only after every artifact has been constructed successfully.
        # Each replacement is atomic; the four replacements are not a transaction.
        os.replace(staged_index, index_path)
        os.replace(staged_chunks, chunks_path)
        os.replace(staged_fts, fts_path)
        os.replace(staged_rag, rag_path)

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunks saved to: {chunks_path}")


if __name__ == "__main__":
    build_index()
