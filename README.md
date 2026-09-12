# Local RAG/MCP Knowledge Base Assistant

A local question-answering assistant over your documents, using hybrid retrieval
and Ollama's `qwen3:0.6b`. The current implementation is complete against
[SPEC.md](SPEC.md): semantic FAISS search and lexical SQLite FTS5 search select
contexts for the existing answer-generation and MCP flow.

## Architecture

Index construction recursively loads `.txt`, `.md`, `.pdf`, and `.docx` files
from the configured documents directory. Text is split with tiktoken's
`cl100k_base` encoding into **700-token chunks with 100-token overlap**.
SentenceTransformers (`all-MiniLM-L6-v2`) generates embeddings, which are
normalized and stored in a FAISS `IndexFlatIP` index. SQLite FTS5 indexes the
text of the same ordered chunks.

```text
Documents → token chunks ─┬→ embeddings → FAISS
                         ├→ chunk dictionaries (chunks.pkl)
                         └→ SQLite FTS5

User query
    ↓
Query Expansion (configured local Qwen through Ollama)
    ↓
Shared artifact readiness (load/rebuild before parallel searches)
    ├→ Vector search: original natural-language query ─┐
    └→ FTS search: original query + expansion terms ───┤ parallel
                                                     ↓
                                         Reciprocal Rank Fusion
                                                     ↓
                                          Up to TOP_K chunks
                                                     ↓
                                  Existing MCP decision/tool flow
                                                     ↓
                                 Context prompt → Ollama → answer
```

`rag.query.retrieve(query)` is the production hybrid retrieval interface. Query
Expansion extracts a bounded set of search terms; malformed output, empty
output, timeout, or model failure falls back to the original query. The vector
branch always embeds the original question. The two search branches run
concurrently, then RRF combines their ranked chunk positions with equal weights
and one-based ranks: `sum(1 / (RRF_K + rank))`. Ties are resolved deterministically.

The result contains up to `TOP_K` chunk dictionaries with `text`, `source`, and
`chunk_id`. Either search branch can fail while the other supplies contexts;
if neither supplies usable contexts, retrieval returns an empty list. This
fallback concerns retrieval; final answer generation still needs Ollama.

`rag.query.retrieve_vector(query)` is a vector-only helper for reproduction and
benchmarking. Application callers continue to use `retrieve(query)`.

## Generated artifacts and paths

| Artifact | Purpose |
| --- | --- |
| `index.faiss` | FAISS vector index |
| `chunks.pkl` | Ordered chunk dictionaries shared by both searches |
| `fts_index.db` | Persistent SQLite FTS5 text index |

All three are generated from the same chunks and can be rebuilt from source
documents. SQLite is derived search data, not canonical document storage.
FAISS and FTS results map to positions in the shared chunk list.

Run application commands from `src`. With defaults, all three artifacts are
written there and documents are read from `src/docs/`. Relative
`FAISS_INDEX_PATH` and `CHUNKS_PATH` values resolve relative to `src`;
`DOCUMENTS_DIR` and `FTS_INDEX_PATH` use normal `Path` semantics, so relative
values resolve from the current working directory. Absolute paths remain absolute.

**Upgrading an old vector-only installation:** run `build-index` again to create
`fts_index.db`. Existing FAISS/chunks artifacts do not trigger a rebuild merely
because the FTS index is missing. Rebuild after changing source documents, too.

## Setup and usage

From the repository root, create the project environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/requirements.txt
```

With Ollama installed and its local service running (these commands are
independent of working directory):

```bash
ollama pull qwen3:0.6b
ollama list
```

From the repository root, enter `src` before building or running:

```bash
cd src
mkdir -p docs
# Add .txt, .md, .pdf, or .docx documents under docs/.
../.venv/bin/python main.py build-index
../.venv/bin/python main.py
```

Type `exit`, `quit`, or `q` to leave the interactive CLI. Models and the tiktoken
encoding need to be downloaded/cached before offline use. See
[src/COMMANDS.md](src/COMMANDS.md) for setup, rebuild, and test commands, and
[src/README.md](src/README.md) for the module layout and troubleshooting.

## Configuration

Current defaults in [src/config.py](src/config.py):

| Setting | Default | Purpose |
| --- | --- | --- |
| `DOCUMENTS_DIR` | `"./docs"` | Recursively ingested source directory |
| `CHUNK_SIZE` | `700` | Tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap in tokens |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | SentenceTransformers embedding model |
| `OLLAMA_MODEL` | `"qwen3:0.6b"` | Local model for expansion and answer/MCP decisions |
| `OLLAMA_URL` | `"http://localhost:11434/api/generate"` | Expansion and answer-generation endpoint |
| `FAISS_INDEX_PATH` | `"index.faiss"` | Vector index path |
| `CHUNKS_PATH` | `"chunks.pkl"` | Shared chunk mapping path |
| `FTS_INDEX_PATH` | `"fts_index.db"` | SQLite FTS5 path, relative to cwd when not absolute |
| `TOP_K` | `5` | Maximum final chunks |
| `VECTOR_CANDIDATES` | `20` | Vector candidate depth before fusion |
| `FTS_CANDIDATES` | `20` | Lexical candidate depth before fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion rank constant |
| `QUERY_EXPANSION_MAX_TERMS` | `5` | Maximum additional search terms/phrases |
| `QUERY_EXPANSION_TEMPERATURE` | `0.0` | Expansion generation temperature |
| `QUERY_EXPANSION_TIMEOUT` | `15` | Expansion HTTP request timeout in seconds |

## MCP compatibility

Hybrid retrieval changes context selection while preserving the existing
assistant and MCP integration. After retrieval, the assistant can ask the local
model whether to call `read_document(file_path)`, `list_documents()`, or
`search_documents(query)` (filename search). Available tool output is added to
the answer prompt, and the CLI displays retrieved sources.

MCP tools execute locally. Document reads resolve both the requested path and
configured documents root, rejecting paths that resolve outside that root,
including sibling-prefix paths and symlink escapes.

## Tests and benchmarks

Run the complete automated suite **from the repository root**:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See [bench/README.md](bench/README.md) for the frozen datasets, vector-versus-hybrid
methodology, retained measurements, and reproduction commands. Those measurements
include Query Expansion in hybrid retrieval latency and are specific to their
recorded runs; they are not end-to-end answer timings.
