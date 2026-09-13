import faiss
from contextlib import contextmanager
import hashlib
import logging
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.ingest import ingest_documents
from rag.chunk import chunk_documents
from rag.embed import embed_chunks
from rag.store import build_rag_db
from config import FAISS_INDEX_PATH, RAG_DB_PATH

logger = logging.getLogger(__name__)

# Published legacy defaults, used only for post-publication cleanup.
_LEGACY_CHUNKS_PATH = "chunks.pkl"
_LEGACY_FTS_DB_PATH = "fts_index.db"


def _sha256_file(path: Path) -> str:
    """Hash the exact written artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Build FAISS and rag.db from the same ordered chunks."""
    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
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
    rag_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        _staged_artifact(index_path) as staged_index,
        _staged_artifact(rag_path) as staged_rag,
    ):
        faiss.write_index(index, str(staged_index))
        faiss_sha256 = _sha256_file(staged_index)
        print("📦 Creating chunk database and FTS index...")
        build_rag_db(chunks, staged_rag, faiss_sha256=faiss_sha256)

        # Publish only after every artifact has been constructed successfully.
        # Each replacement is atomic; the two replacements are not a transaction.
        os.replace(staged_index, index_path)
        os.replace(staged_rag, rag_path)

    # Preserve the old src-relative chunks and cwd-relative FTS locations.
    for legacy_path in (src_dir / _LEGACY_CHUNKS_PATH, Path(_LEGACY_FTS_DB_PATH)):
        try:
            legacy_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove legacy file %s", legacy_path, exc_info=True)

    print(f"✅ Indexing complete: {len(chunks)} chunks indexed")
    print(f"   Index saved to: {index_path}")
    print(f"   Chunk database saved to: {rag_path}")


if __name__ == "__main__":
    build_index()
