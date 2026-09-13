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
retrieve(query: str) -> list[dict]
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

## 4. Retrieval Storage and Indexes

The application uses two independent retrieval indexes over the same chunks.

### 4.1 Vector Index

FAISS remains the vector search engine.

Existing embedding behavior must remain compatible with the current application.

### 4.2 SQLite Chunk Store and Full-Text Index

Store generated chunks and a persistent SQLite FTS5 external-content index in `rag.db`.

The FTS index:

* is built from the same chunks used by FAISS;
* contains searchable chunk text;
* preserves a stable mapping back to the original chunk;
* is derived data and can be rebuilt;
* must not become the canonical storage for source documents.

The expected generated artifacts are:

```text
index.faiss
rag.db
```

Source documents remain the canonical source data outside SQLite. Within a retrieval generation, `rag.db` is the canonical persisted store for generated chunk data; it remains derived, rebuildable data, not canonical source-document storage.

`rag.db` contains the following content table and external-content FTS index over `chunks.text`:

```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    chunk_id INTEGER NOT NULL
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='unicode61'
);
```

The required global retrieval identity invariant is:

```text
FAISS vector position == chunks.id == chunks_fts.rowid
```

For N generated chunks, global retrieval IDs form the contiguous range `0..N-1` and the FAISS index contains exactly N vectors.

Global retrieval IDs are assigned explicitly from the zero-based positions in the ordered chunk list. ID `0` is valid and must be searchable and resolvable, including through FTS `MATCH`. The existing `chunk_id` field remains the per-source-document chunk ordinal and is not the global retrieval identity. Returned chunk dictionaries preserve `text`, `source`, and `chunk_id` from the corresponding `chunks` row.

### 4.3 External-Content FTS Lifecycle

Every full `rag.db` build/rebuild must perform all SQLite chunk and FTS construction in one transaction:

1. create the `chunks` table;
2. create the external-content `chunks_fts` table;
3. insert the complete ordered chunk set into `chunks`, with explicit zero-based IDs;
4. explicitly build the real FTS index from the content table using the FTS5 rebuild command:

   ```sql
   INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');
   ```

5. commit only after both the generated chunk rows and the searchable FTS index are complete.

Inserting rows into `chunks` alone does not constitute a completed FTS build. Reading content rows through the external-content table is not proof that a real `MATCH` search works. Construction failure must roll back the SQLite transaction, including schema construction.

A published generated `rag.db` is not mutated incrementally. Changes require building a new full retrieval generation; SQLite synchronization triggers are not required.

### 4.4 Generation Construction and Publication

A retrieval generation consists of `index.faiss` and `rag.db`, both built from the same exact ordered chunk list. Completely construct both artifacts in staged sibling files of their respective final paths before publication of either final artifact begins. The staged `rag.db` must already contain the complete `chunks` table and a fully built, searchable external-content FTS index, with its construction transaction committed.

Any construction failure must leave the previous published generation untouched. Final publication may use separate atomic file replacements; cross-file transactional or crash-atomic publication is not required. This does not relax the construction-failure integrity guarantee or the prohibition on mixed-generation ID mapping in Section 8.

### 4.5 Legacy Generated Artifacts

`chunks.pkl` and the old standalone `fts_index.db` are legacy generated artifacts. After migration they must never participate in runtime readiness, fallback, chunk mapping, FTS, or rebuild decisions, even if they exist and contain conflicting data.

After successful publication of both new final artifacts, remove legacy generated files on a best-effort basis. Log cleanup failures without invalidating the new generation, rolling back publication, or making legacy files runtime inputs. Legacy files may physically remain after cleanup failure and must still be ignored.

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

Healthy requests must also remain capable of overlapping vector/FTS retrieval across requests; generation readiness and rebuild safety must not globally serialize healthy searches.

Each search leg returns an ordered candidate list.

The candidate depth used before fusion may be larger than the final `TOP_K`.

## 8. Failure Isolation

The two search legs are independent.

Required behavior:

* query expansion failure → search using the original query;
* FTS index/build or search failure while `rag.db` chunk storage remains readable → continue with vector results mapped through chunks from the same generation;
* FAISS loading, vector model, or vector search failure while valid `rag.db` exists → continue with FTS results resolved through that database when available, including when an attempted rebuild fails;
* canonical generated chunk storage in `rag.db` unavailable or corrupt → FAISS IDs cannot safely resolve to public chunk dictionaries; return no contexts if safe resolution cannot be restored;
* one empty search result → use the other result list;
* both searches unavailable or empty → return no contexts without crashing the process.

Failures should be observable through logging but must not corrupt generated indexes or terminate the application unnecessarily.

A request must never map IDs from one retrieval generation through chunk data from another. Search results and their chunk mapping must remain tied to the same generation throughout search, fusion, and resolution. Unrelated or stale storage must not substitute for unavailable chunk storage.

Preserve rebuild/fallback race safety: prepare a coherent generation before searching; after a successful rebuild, use that generation's chunk mapping even if FAISS loading fails. If the new chunk storage cannot be read, do not reuse an older cached mapping for new IDs. A fallback request following failed repair must remain safe from another request rebuilding or replacing its generation while it searches and resolves chunks. These are behavioral invariants and do not prescribe a particular lock implementation or require cross-process rebuild coordination.

Preserve lazy `SentenceTransformer` initialization as vector-leg work. Importing retrieval code and preparing shared storage must not eagerly construct the query model. Concurrent initialization must reuse one successfully initialized model; a failed initialization must remain retryable on later requests and allow FTS fallback for the current request. Encoding or vector search failures must likewise preserve FTS fallback.

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

Search weights may be configurable, but the default weights remain equal.

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
RAG_DB_PATH
QUERY_EXPANSION_MAX_TERMS
QUERY_EXPANSION_TEMPERATURE
```

Reasonable defaults must work without additional user configuration.

`RAG_DB_PATH` configures the combined generated chunk and FTS database, defaulting to `rag.db`; `FAISS_INDEX_PATH` continues to configure `index.faiss`. The final storage configuration no longer requires `CHUNKS_PATH` or a separate `FTS_INDEX_PATH`. Other retrieval configuration requirements remain unchanged, including `TOP_K = 5` and default `RRF_K = 60`.

## 13. Tests

Automated tests must cover the retrieval logic without requiring external network access.

Required test areas:

### Generated chunk storage and publication

* `chunks` schema and round-trip public chunk dictionaries;
* zero-based global IDs, specifically ID `0`, distinct from per-source `chunk_id`;
* `FAISS vector position == chunks.id == chunks_fts.rowid` for the same exact ordered chunk list;
* transactional SQLite schema, chunk, and FTS construction, including rollback on failure;
* both staged artifacts complete, with real searchable FTS, before either final artifact is published;
* failed staged generation construction preserving the previous published generation;
* legacy `chunks.pkl` and `fts_index.db` ignored for runtime readiness, fallback, chunk mapping, FTS, and rebuild decisions, even when present with conflicting data;
* legacy cleanup only after both final artifacts are published, with cleanup failure logged and leaving the new generation valid and legacy files ignored.

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

Tests must exercise real `MATCH` queries against the explicitly rebuilt external-content FTS index, including `chunks_fts.rowid == chunks.id == 0`. Inserting chunks without the explicit FTS rebuild must not be treated as proof of a completed FTS generation; content-table reads alone are insufficient.

### RRF

* rank calculation;
* documents found by one search;
* documents found by both searches;
* deterministic ordering;
* final top-K truncation.

### Hybrid retrieval

* vector and FTS result fusion;
* FTS failure with readable chunks preserving vector fallback;
* FAISS/vector failure with valid `rag.db` preserving FTS fallback, including failed rebuilds;
* unavailable or corrupt chunk storage preventing unsafe ID resolution;
* no mixed-generation ID-to-chunk mapping, including stale cached mappings after rebuild;
* concurrent vector/FTS execution within and across healthy requests;
* rebuild/fallback race safety with overlapping requests;
* lazy vector model initialization, concurrent initialization, failure retry, and FTS fallback;
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

### Storage Migration Regression Contract

Preserve strict regression comparison against the retained historical results. Frozen inputs and retained results must not be edited:

* evaluation corpus: `src/docs/evaluation`;
* challenge corpus: `src/docs/challenge`;
* query files and expected labels: `bench/queries.json` and `bench/challenge_queries.json`;
* challenge freeze manifest: `bench/challenge-inputs.sha256`;
* retained results: `bench/results/vector-baseline.json`, `bench/results/vector-challenge-baseline.json`, `bench/results/hybrid-sanity.json`, and `bench/results/hybrid-challenge.json`.

Compare each post-migration vector-only and hybrid run on both corpora with its corresponding retained historical result, preserving exactly:

* ordered query IDs;
* per-query `ranked_sources` and `expected_source_rank`;
* per-query hit (`hit_at_5`) and reciprocal rank (`reciprocal_rank`);
* overall Hit@5 and MRR@5;
* per-category Hit@5 and MRR@5;
* additionally for hybrid runs, per-query `expansion_inputs`.

Latency equality is explicitly not required. New measurements must not overwrite retained historical result files.

If rankings differ after the storage migration, treat the difference as a regression to investigate unless an independently identified storage bug proves otherwise. Do not tune prompts, queries, weights, candidate depths, corpora, labels, or benchmark fixtures merely to restore historical metrics.

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

This storage change moves only generated chunk persistence into SQLite; source documents remain canonical source data outside SQLite. It does not include replacing FAISS, changing embedding behavior, query expansion, RRF/retrieval semantics, MCP, or final answer generation. It also excludes incremental mutation of published `rag.db`, SQLite synchronization triggers, cross-process rebuild coordination, cross-file transactional/crash-atomic publication, and benchmark tuning.

## 16. Completion Criteria

The implementation is complete when:

1. supported documents can be ingested and indexed successfully;
2. the only final generated retrieval artifacts are `index.faiss` and `rag.db`, built from the same exact ordered chunk list with `FAISS vector position == chunks.id == chunks_fts.rowid`, including ID `0`;
3. query expansion runs before retrieval;
4. malformed query-expansion output falls back safely;
5. vector and FTS searches execute concurrently;
6. both ranked lists are fused with RRF;
7. only final `TOP_K` chunks are passed to the existing answer flow;
8. either retrieval leg can fail without unnecessarily failing the whole query;
9. existing callers of `retrieve()` remain compatible;
10. automated retrieval tests pass;
11. vector-only and hybrid retrieval preserve the strict historical regression contract in Section 14;
12. SQLite chunk and external-content FTS construction completes in one transaction, including explicit FTS rebuild and real `MATCH` searchability, before either staged artifact is published;
13. failed staged construction preserves the previous published generation, and retrieval preserves failure isolation, generation-consistent mapping, rebuild/fallback race safety, healthy concurrency, and lazy vector model retry;
14. `chunks.pkl` and standalone `fts_index.db` are absent from the final storage architecture and are not runtime dependencies; best-effort cleanup occurs only after both new artifacts are published, and any remaining legacy files are ignored.
