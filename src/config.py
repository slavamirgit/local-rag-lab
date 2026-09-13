# Configuration for Company Knowledge Base Assistant

# Document directory (relative paths resolve from the current working directory)
DOCUMENTS_DIR = "./docs"

# Token chunking configuration (cl100k_base tokens, not characters)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# FAISS index path (relative to src directory)
FAISS_INDEX_PATH = "index.faiss"

# Lexical candidate count before fusion
FTS_CANDIDATES = 20

# Generated chunk database (normal Path semantics; relative paths resolve from cwd)
RAG_DB_PATH = "rag.db"

# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:0.6b"

# Query expansion configuration
QUERY_EXPANSION_MAX_TERMS = 5
QUERY_EXPANSION_TEMPERATURE = 0.0
QUERY_EXPANSION_TIMEOUT = 15

# RAG retrieval configuration
TOP_K = 5
VECTOR_CANDIDATES = 20
RRF_K = 60
