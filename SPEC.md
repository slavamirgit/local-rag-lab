# Local RAG — Hybrid Retrieval Specification

## 1. Purpose

The application provides retrieval-augmented question answering over a local document collection.

Retrieval must combine:

- semantic vector search;
- lexical full-text search;
- LLM-assisted query expansion.

The resulting ranked chunks are passed to the existing answer-generation flow and MCP integration.

## 2. Existing Components

The application already provides:

- document ingestion;
- document chunking;
- embeddings generated with `sentence-transformers`;
- FAISS vector indexing;
- local inference through Ollama;
- Qwen as the local language model;
- MCP tools;
- final answer generation from retrieved contexts.

The public retrieval interface is:

```python
retrieve(query: str)
```

It returns ranked chunk dictionaries containing at least:

```python
{
    "text": str,
    "source": str,
    "chunk_id": int,
}
```

Existing callers must continue to use this interface.

## 3. Document Ingestion

Supported document types remain:

* `.txt`
* `.md`
* `.pdf`
* `.docx`

Document discovery must work recursively under the configured documents directory.

Running the index build command must complete successfully when valid documents are present.

## 4. Indexes

The application uses two independent retrieval indexes over the same chunks.

### 4.1 Vector Index

FAISS remains the vector search engine.

Existing embedding behavior must remain compatible with the current application.

### 4.2 Full-Text Index

Add a persistent SQLite FTS5 index.

The FTS index:

* is built from the same chunks used by FAISS;
* contains searchable chunk text;
* preserves a stable mapping back to the original chunk;
* is derived data and can be rebuilt;
* must not become the canonical storage for source documents.

The expected generated artifacts are:

```text
index.faiss
chunks.pkl
fts_index.db
```

A chunk identifier used by FTS must resolve unambiguously to the same chunk represented by the corresponding FAISS result.

## 5. Query Expansion

Before retrieval, the application asks the configured local Qwen model to extract a small set of useful search terms from the user's question.

The model call must:

* use a short and explicit prompt;
* request only search terms or short search phrases;
* use deterministic generation (`temperature = 0` or equivalent);
* avoid requiring free-form explanations;
* return a small bounded number of terms.

The implementation must tolerate malformed model output.

If query expansion fails because of:

* model unavailability;
* timeout;
* invalid output;
* empty output;
* parsing error;

retrieval must continue using the original user query.

Query expansion failure must never make the user query fail.

## 6. Search Inputs

The semantic and lexical retrieval legs have different inputs.

### Vector search

Vector search uses the original natural-language user query.

### Full-text search

Full-text search uses:

* the original query;
* the generated search terms.

The original query must always remain part of lexical retrieval even when query expansion succeeds.

## 7. Parallel Retrieval

Vector search and full-text search must execute concurrently.

The implementation may use:

```python
ThreadPoolExecutor
```

or an equivalent concurrency mechanism suitable for the existing synchronous codebase.

The two independent searches must not be executed sequentially.

Each search leg returns an ordered candidate list.

The candidate depth used before fusion may be larger than the final `TOP_K`.

## 8. Failure Isolation

The two search legs are independent.

Required behavior:

* query expansion failure → search using the original query;
* FTS failure → continue with vector results;
* vector search failure → continue with FTS results when available;
* one empty search result → use the other result list;
* both searches unavailable or empty → return no contexts without crashing the process.

Failures should be observable through logging but must not corrupt generated indexes or terminate the application unnecessarily.

## 9. Reciprocal Rank Fusion

Vector and FTS rankings are combined using Reciprocal Rank Fusion.

For a document `d`:

```text
RRF(d) = Σ 1 / (k + rank(d))
```

Use one-based ranks.

Default:

```text
k = 60
```

A document present in both result lists receives contributions from both lists.

The fusion implementation must:

* operate only on ranked chunk identifiers;
* not depend on FAISS or SQLite internals;
* produce deterministic ordering;
* support final truncation to `TOP_K`.

Search weights may be configurable, but the default behavior must support equal weights.

## 10. Hybrid Retrieval Pipeline

The retrieval pipeline is:

```text
User query
    |
    v
Query Expansion
    |
    +-------------------------+
    |                         |
    v                         v
Vector Search             Full-Text Search
(original query)          (query + terms)
    |                         |
    +------------+------------+
                 |
                 v
        Reciprocal Rank Fusion
                 |
                 v
              Top-K
                 |
                 v
        Existing answer flow
```

`retrieve(query)` remains the public retrieval boundary.

Callers must not need to know whether contexts came from vector search, FTS, or both.

## 11. Final Answer Generation

The existing prompt-building and answer-generation flow remains responsible for producing the final answer.

Hybrid retrieval changes context selection only.

MCP behavior must remain compatible with the existing application.

## 12. Configuration

Retrieval-specific constants must be configurable rather than scattered through implementation code.

Configuration should cover at least:

```text
TOP_K
VECTOR_CANDIDATES
FTS_CANDIDATES
RRF_K
FTS_INDEX_PATH
QUERY_EXPANSION_MAX_TERMS
QUERY_EXPANSION_TEMPERATURE
```

Reasonable defaults must work without additional user configuration.

## 13. Tests

Automated tests must cover the retrieval logic without requiring external network access.

Required test areas:

### Query expansion

* valid model output;
* malformed output;
* empty output;
* model exception;
* fallback to original query;
* maximum term count.

### Full-text search

* index creation;
* exact-term retrieval;
* deterministic ranking;
* empty query;
* missing or unavailable index behavior;
* mapping FTS results back to the correct chunks.

### RRF

* rank calculation;
* documents found by one search;
* documents found by both searches;
* deterministic ordering;
* final top-K truncation.

### Hybrid retrieval

* vector and FTS result fusion;
* FTS failure fallback;
* vector failure fallback;
* query-expansion fallback;
* empty search results;
* preservation of the public `retrieve()` result format.

External LLM and embedding calls should be replaceable or injectable in tests.

## 14. Evaluation

Provide a reproducible comparison between:

```text
vector-only retrieval
```

and:

```text
hybrid retrieval
```

The evaluation dataset should contain queries representing:

* exact technical terms;
* filenames or commands;
* natural-language semantic questions;
* mixed lexical/semantic queries.

Useful retrieval metrics include:

* Hit@K;
* Mean Reciprocal Rank (MRR);
* retrieval latency.

Evaluation must report actual measured results rather than assume that hybrid retrieval is always superior.

## 15. Non-Goals

The implementation does not require:

* replacing FAISS;
* replacing the existing embedding model;
* changing the MCP protocol;
* rewriting the answer-generation layer;
* introducing a remote vector database;
* migrating canonical document storage into SQLite;
* adding a web interface;
* redesigning document chunking unless testing demonstrates a concrete need.

## 16. Completion Criteria

The implementation is complete when:

1. supported documents can be ingested and indexed successfully;
2. FAISS and SQLite FTS5 indexes are created from the same chunk set;
3. query expansion runs before retrieval;
4. malformed query-expansion output falls back safely;
5. vector and FTS searches execute concurrently;
6. both ranked lists are fused with RRF;
7. only final `TOP_K` chunks are passed to the existing answer flow;
8. either retrieval leg can fail without unnecessarily failing the whole query;
9. existing callers of `retrieve()` remain compatible;
10. automated retrieval tests pass;
11. vector-only and hybrid retrieval can be compared with a reproducible evaluation script.
