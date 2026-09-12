# Frozen vector retrieval datasets

Both synthetic datasets were defined before hybrid retrieval was implemented.
Each corpus and its query labels were frozen before its first vector retrieval
run. Neither dataset was tuned after observing results. They are separate inputs
for future comparisons, not a combined index.

| Dataset | Documents | Queries | Inputs | Recorded result |
| --- | ---: | ---: | --- | --- |
| Sanity | 15 | 15 | `src/docs/evaluation/`, `bench/queries.json` | `bench/results/vector-baseline.json` |
| Challenge | 24 | 18 | `src/docs/challenge/`, `bench/challenge_queries.json` | `bench/results/vector-challenge-baseline.json` |

The original sanity set covers distinct internal knowledge-base topics. It has
five queries per category and achieved Hit@5=1.0 and MRR@5=1.0. Its documents,
queries, and recorded result remain byte-for-byte unchanged. It is useful for
basic regressions but has no accuracy headroom on these metrics.

The challenge set has six queries per category and three clusters of eight
similar runbooks. Queue notices distinguish near-identical status identifiers
and required actions. Relay procedures distinguish filenames, configuration
keys, service names, and versioned resources. Recovery procedures distinguish
command flags and short hold codes. Six documents serve as additional
distractors without being the expected source of a query. Semantic queries
paraphrase the required operation without copying its identifier; mixed queries
combine an exact detail with a procedural question.

Both query files use stable IDs, `exact_lexical`, `semantic`, or `mixed`
categories, and repository-relative expected source paths. Each query has one
unambiguous expected document. Keep the inputs and labels frozen; create a
separately versioned dataset if further cases are needed.

The retained `bench/challenge-inputs.sha256` was written before the first
challenge retrieval run. The runner verifies it before loading models and
records its own hash plus all input hashes in the measured result. Verify it
from the repository root with:

```bash
sha256sum -c bench/challenge-inputs.sha256
```

## Reproduction

From the repository root, using the project virtual environment and its cached
`all-MiniLM-L6-v2` model:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python bench/vector_baseline.py --dataset challenge --output /tmp/local-rag-challenge-repeat.json
HF_HUB_OFFLINE=1 .venv/bin/python bench/vector_baseline.py --dataset sanity --output /tmp/local-rag-sanity-repeat.json
```

The project dependencies in `src/requirements.txt` must be installed. Offline
execution requires the embedding model and tiktoken encoding to be cached.
On a fresh installation, omit `HF_HUB_OFFLINE=1` to permit the normal model
download. Match the package versions and model revision recorded in the result
metadata when reproducing this measurement.

Use `--output` for subsequent measurements to preserve the frozen result files.
Without it, the runner writes the selected dataset's recorded-result path shown
in the table. Omitting `--dataset` selects sanity, preserving the original CLI
default. The original sanity result retains the historical runner fingerprint;
the challenge result fingerprints the shared runner after dataset selection was
added. The original result was not regenerated during this addition.

The shared script always rebuilds the selected index through `rag.build_index`, then
calls the existing `rag.query.retrieve` once for each query in file order. Only
document and artifact paths are overridden in process. Documents are sorted by
path before the production index builder runs. Seeds are fixed to zero and
PyTorch uses one CPU thread to reduce run-to-run scheduling variation. The model,
chunking, TOP_K=5, normalization, and FAISS IndexFlatIP behavior are unchanged.
The script never calls answer generation or Ollama.

Runtime artifacts are written to `bench/.runtime/<dataset>/index.faiss` and
`bench/.runtime/<dataset>/chunks.pkl`, where dataset is `sanity` or `challenge`.
They remain ignored, disposable build products and are separate from each other
and the normal application index. Earlier sanity artifacts may remain directly
under `bench/.runtime/`; the shared runner does not use them. It verifies exactly
one chunk per source. Sanity documents contain 57–75 cl100k_base tokens and
challenge documents contain 67–78, below the current 700-token chunk size and
600-token step (100-token overlap).

Each result records input and implementation SHA-256 hashes,
model revision, package versions, settings, index order, and all ranked sources
and query latencies. Reproducing the challenge run returned identical ranked
source ordering for all 18 queries. Wall-clock latencies varied as expected.

## Measurement

Hit@5 is the fraction of queries whose expected source appears in the five
returned chunks. MRR@5 is the mean reciprocal rank of that source, with zero for
a miss. Retrieval only exposes five results, so this is a cutoff MRR, not a
measurement of ranks beyond five. Each query has one relevant source and each
source has one chunk; returned sources are not deduplicated.

Latency uses `time.perf_counter` around `retrieve()` after one unmeasured warm-up
query (`internal knowledge base`). It includes query embedding, normalization,
FAISS search, and result assembly. Index construction/loading, model loading,
warm-up, scoring, and answer generation are excluded. Each query is measured
once; mean, median, and nearest-rank p95 are recorded overall and by category.
Timing is machine-dependent and this small sample is not a load test.

The original sanity measurement has 10.95 ms overall mean retrieval latency on
CPU. The first challenge measurement on CPU is:

| Category | Queries | Hit@5 | MRR@5 | Mean latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| All | 18 | 0.944444 | 0.736111 | 11.28 |
| exact_lexical | 6 | 1.000000 | 0.652778 | 9.40 |
| semantic | 6 | 1.000000 | 0.805556 | 12.27 |
| mixed | 6 | 0.833333 | 0.750000 | 12.17 |

The mixed query `challenge-mix-01` missed the expected source in the top five.
Three exact lexical, two semantic, and one other mixed query found the expected
source below rank one. Inputs and labels were retained unchanged after these
observations. These are vector-only measurements; no hybrid-search benefit has
been measured or established. Future comparisons must use the same inputs,
labels, cutoff, and timing protocol and label the implementation actually
measured. The runner calls the production implementation directly and does not
retain a separate copy of its vector algorithm.
