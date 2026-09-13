# Setup and run commands

These commands use a `.venv` in the **repository root**. Application commands
run from **`src`**. Benchmark and automated test commands run from the
**repository root**.

## Prerequisites

Install Python, Ollama, and use a Python SQLite build with FTS5 support. The
existing automated suite uses `contextlib.chdir` and requires Python 3.11+.

Start the Ollama service if it is not already running (in a separate terminal;
working directory does not matter):

```bash
ollama serve
```

Pull and check the configured model (from any working directory):

```bash
ollama pull qwen3:0.6b
ollama list
```

## Install dependencies

Starting in the **repository root**:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r src/requirements.txt
```

Initial model use may download the embedding model and tiktoken encoding;
cache them before offline operation.

## Prepare documents and build the indexes

Starting in the **repository root**:

```bash
cd src
mkdir -p docs
# Add .txt, .md, .pdf, or .docx files under docs/ (subdirectories are supported).
# Review config.py if using a different documents directory.
../.venv/bin/python main.py build-index
```

This loads documents recursively, splits them into 700-token chunks with
100-token overlap, generates embeddings, and builds:

- `index.faiss`: FAISS vector index.
- `rag.db`: generated chunk dictionaries and an external-content SQLite FTS5 index.

Both artifacts are built from the same ordered chunk set.

The equivalent direct build command, **while in `src`**, is:

```bash
../.venv/bin/python -m rag.build_index
```

Use this build command to rebuild both artifacts. Readiness may also attempt a
build when SQLite chunks or FAISS cannot be loaded.

Relative `FAISS_INDEX_PATH` values resolve relative to `src`.
`DOCUMENTS_DIR` (default `./docs`) and `RAG_DB_PATH` (default `rag.db`)
resolve relative to the current working directory when not absolute. With
`src` as cwd, the defaults put both artifacts beside `main.py`.
See [README.md](README.md#configuration) for all current configuration defaults.

## Run the interactive application

**While in `src`** after the build:

```bash
../.venv/bin/python main.py
```

Ask a documentation question. The production path uses Query Expansion,
shared artifact readiness, parallel vector/FTS search, RRF, and the existing
answer-generation/MCP flow. Type `exit`, `quit`, or `q` to stop.

## Update the knowledge base

After adding or editing source documents, rebuild **from `src`**:

```bash
../.venv/bin/python main.py build-index
```

Restart a running assistant after rebuilding so it loads the new artifacts.

## Run automated tests

From the **repository root** (run `cd ..` first if currently in `src`):

```bash
.venv/bin/python -m unittest discover -s tests -v
```

For benchmark methodology and reproduction commands, see
[../bench/README.md](../bench/README.md). Benchmark commands run from the repository
root; an explicit `--output` is required. Use new paths outside retained
`bench/results/*.json`, such as `bench/.runtime/storage-regression/`, then compare
candidates with `bench/storage_regression.py` as documented there.

## Troubleshooting

- **No documents found:** check `DOCUMENTS_DIR` and run the build from `src`.
- **Missing indexes or vector-only results:** rebuild both
  artifacts with the build command above. Check warnings, path permissions,
  and SQLite FTS5 support if lexical search remains unavailable.
- **Ollama connection errors:** ensure the service is running, use `ollama list`
  to check availability, and verify `qwen3:0.6b` is installed. Expansion failures
  fall back to the original query; final answer generation still needs Ollama.
  `OLLAMA_URL` defaults to `http://localhost:11434/api/generate` for these HTTP calls.
- **MCP client initialization errors:** the assistant can continue without MCP tools.
- **Import errors:** use the project environment and the specified cwd; reinstall
  dependencies from the repository root if needed.
