# Company Knowledge Base Assistant

A local Q&A system using hybrid RAG (Retrieval-Augmented Generation), Ollama's
`qwen3:0.6b`, and the existing MCP (Model Context Protocol) document tools.

## Features

- Recursive ingestion of `.txt`, `.md`, `.pdf`, and `.docx` documents.
- Token chunking with tiktoken (`cl100k_base`), 700 tokens per chunk and 100-token overlap.
- SentenceTransformers embeddings and FAISS semantic search.
- Persistent SQLite FTS5 lexical search over the same chunks.
- Local Qwen Query Expansion, parallel vector/FTS searches, and Reciprocal Rank Fusion.
- Retrieval fallback when expansion or an individual search branch fails.
- Local answer generation with source attribution and optional MCP document access.

## Setup and usage

From the **repository root**, install dependencies into the project environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/requirements.txt
```

Install Ollama and start its local service, then pull the configured model
(from any working directory):

```bash
ollama pull qwen3:0.6b
```

Application commands below run **from `src`**, using the environment in the
repository root:

```bash
# Starting at the repository root:
cd src
mkdir -p docs
# Add your documents under docs/ and review config.py.
../.venv/bin/python main.py build-index
../.venv/bin/python main.py
```

Type `exit`, `quit`, or `q` to stop the CLI. The supported direct build entry
point, also from `src`, is `../.venv/bin/python -m rag.build_index`.
See [COMMANDS.md](COMMANDS.md) for the command reference.

## Project structure

```text
local-rag-lab/
├── SPEC.md                   # Hybrid retrieval specification
├── .venv/                    # Project environment (local setup)
├── tests/                    # Automated tests
├── bench/                    # Frozen datasets/measurements and benchmark runners
└── src/
    ├── config.py             # Application and retrieval settings
    ├── main.py               # build-index and interactive CLI
    ├── assistant.py          # Retrieval, MCP decision, and answer orchestration
    ├── rag/
    │   ├── ingest.py         # Recursive document ingestion
    │   ├── chunk.py          # Token chunking
    │   ├── embed.py          # SentenceTransformers embeddings
    │   ├── build_index.py    # Build FAISS, chunk mapping, and FTS artifacts
    │   ├── fts.py            # SQLite FTS5 construction and lexical search
    │   ├── query_expansion.py # Local Qwen search-term extraction and fallback
    │   ├── fusion.py         # Deterministic Reciprocal Rank Fusion
    │   └── query.py          # Hybrid retrieval, vector helper, prompt/answer helpers
    ├── mcp/
    │   ├── server.py         # Document tool definitions
    │   └── client.py         # MCP subprocess client
    ├── docs/                 # Default source document directory
    ├── index.faiss           # Generated vector index
    ├── chunks.pkl            # Generated ordered chunk dictionaries
    ├── fts_index.db          # Generated lexical index (with src as cwd)
    ├── requirements.txt
    ├── COMMANDS.md
    └── README.md
```

## Indexing and path resolution

`build-index` loads documents, creates token chunks, generates embeddings, and
builds FAISS and SQLite FTS5 indexes from the same ordered chunk set. The three
generated artifacts are `index.faiss`, `chunks.pkl`, and `fts_index.db`.
FTS row IDs map back to positions in `chunks.pkl`, matching FAISS positions;
the per-document `chunk_id` field is not a globally unique index ID. Source
files remain the canonical documents.

Relative `FAISS_INDEX_PATH` and `CHUNKS_PATH` values resolve relative to `src`.
`DOCUMENTS_DIR` and `FTS_INDEX_PATH` use normal `Path` semantics: relative values
resolve from the current working directory. Absolute paths remain absolute.
Running builds and the application from `src` keeps the default documents and
all three artifacts in the locations shown above.

Readiness can attempt an automatic build if chunks or FAISS cannot be loaded.
**An installation with old FAISS/chunks artifacts must run `build-index` again
to create `fts_index.db`.** A missing FTS index alone does not trigger that
rebuild; lexical search logs its failure and retrieval uses available vector
results. Rebuild after adding or updating documents, then restart a running
assistant so it loads the rebuilt artifacts.

## Retrieval and answer flow

1. `retrieve(query)` asks the configured local Qwen model for a bounded set of
   search terms. The original query is retained; unusable output, timeout, or
   model failure falls back to that query alone.
2. Shared artifact readiness loads/rebuilds artifacts before parallel searches.
3. Two searches run concurrently: vector search embeds the **original
   natural-language query**, and FTS5 searches the **original query plus expansion
   terms**, returning candidates ordered by BM25 with deterministic tie-breaking.
4. RRF combines the candidate rankings using equal weights and one-based ranks,
   summing `1 / (RRF_K + rank)` for each chunk. Ties use chunk position.
5. Retrieval returns up to `TOP_K` dictionaries containing `text`, `source`, and
   `chunk_id`. If one branch fails or is empty, the other can supply contexts;
   if neither supplies usable results, retrieval returns an empty list.
6. The existing assistant decides whether to use MCP tools, builds the context
   prompt, appends any tool output, and asks Ollama for the final answer.

`rag.query.retrieve(query)` is the **production hybrid path** used by the
application. `rag.query.retrieve_vector(query, limit=None)` is a **vector-only
helper for reproduction/benchmarking**, defaulting to `TOP_K`. It shares the
vector implementation but does not call Query Expansion, FTS search, or RRF.

## Configuration

Edit [config.py](config.py); current defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `DOCUMENTS_DIR` | `"./docs"` | Recursive source directory, relative to cwd |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `100` | Chunk length / overlap in tokens |
| `EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | SentenceTransformers model |
| `OLLAMA_MODEL` | `"qwen3:0.6b"` | Expansion, MCP decisions, and final answers |
| `OLLAMA_URL` | `"http://localhost:11434/api/generate"` | Expansion and answer HTTP endpoint |
| `FAISS_INDEX_PATH` / `CHUNKS_PATH` | `"index.faiss"` / `"chunks.pkl"` | Paths relative to src when not absolute |
| `FTS_INDEX_PATH` | `"fts_index.db"` | Lexical index path, relative to cwd when not absolute |
| `TOP_K` | `5` | Maximum final chunks |
| `VECTOR_CANDIDATES` / `FTS_CANDIDATES` | `20` / `20` | Candidate depths before fusion |
| `RRF_K` | `60` | RRF rank constant |
| `QUERY_EXPANSION_MAX_TERMS` | `5` | Maximum additional terms/phrases |
| `QUERY_EXPANSION_TEMPERATURE` | `0.0` | Expansion temperature |
| `QUERY_EXPANSION_TIMEOUT` | `15` | Expansion HTTP request timeout in seconds |

## MCP tools

The existing server provides `read_document(file_path)` (UTF-8 file reading),
`list_documents()`, and `search_documents(query)` (case-insensitive filename
search). Hybrid retrieval preserves their integration. The document-read path
check does not guarantee strict containment; the pre-existing sibling-path
defect is tracked separately.

## Troubleshooting

- **Missing/stale indexes or FTS failures:** from `src`, run
  `../.venv/bin/python main.py build-index`. Check all three artifact paths,
  permissions, and SQLite FTS5 availability. Restart the assistant after rebuilding.
- **Ollama not responding:** ensure the service is running, check `ollama list`,
  and install `qwen3:0.6b` with `ollama pull qwen3:0.6b`. Check `OLLAMA_URL` for
  expansion and answer calls. Expansion can fall back without Ollama, but final
  answer generation still requires it.
- **No documents found:** check `DOCUMENTS_DIR`, your working directory (`src`),
  and the supported file extensions. Discovery includes nested directories.
- **Empty/partial retrieval:** inspect warnings for chunk loading, vector model/index,
  or FTS failures. One available branch can return results; no usable branches
  yield no contexts.
- **Model/encoding download errors:** initial setup needs the embedding model and
  tiktoken encoding cached before offline use.

## Tests and measurements

From the **repository root** (run `cd ..` first if currently in `src`):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

See [../bench/README.md](../bench/README.md) for vector-versus-hybrid methodology,
frozen inputs, retained measurements, and reproduction commands.

## License

MIT
