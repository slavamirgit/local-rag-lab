"""Rebuild and measure vector-only retrieval on a frozen sanity or challenge corpus."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "sanity": ("src/docs/evaluation", "bench/queries.json", "vector-baseline.json"),
    "challenge": (
        "src/docs/challenge", "bench/challenge_queries.json", "vector-challenge-baseline.json"
    ),
}
CHALLENGE_MANIFEST = ROOT / "bench/challenge-inputs.sha256"
PROTECTED_RESULT_PATHS = (
    ROOT / "bench/results/vector-baseline.json",
    ROOT / "bench/results/vector-challenge-baseline.json",
    ROOT / "bench/results/hybrid-sanity.json",
    ROOT / "bench/results/hybrid-challenge.json",
)


def validate_output_path(output):
    """Require an explicit destination distinct from all retained historical paths."""
    if output is None:
        raise ValueError("An explicit output destination is required; use --output")
    resolved = output.resolve()
    for historical in PROTECTED_RESULT_PATHS:
        if resolved == historical.resolve():
            raise ValueError(
                f"Refusing historical result destination {historical}; "
                "use --output with a different path"
            )
    return output


def fingerprint(paths):
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def aggregate(rows):
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "query_count": len(rows),
        "hit_at_5": statistics.mean(row["hit_at_5"] for row in rows),
        "mrr_at_5": statistics.mean(row["reciprocal_rank"] for row in rows),
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": latencies[math.ceil(0.95 * len(latencies)) - 1],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="sanity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    corpus_name, queries_name, _ = DATASETS[args.dataset]
    corpus = ROOT / corpus_name
    queries_path = ROOT / queries_name
    output = validate_output_path(args.output)

    files = sorted(corpus.glob("*.txt"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    sources = {path.relative_to(ROOT).as_posix() for path in files}
    if len(files) < 5 or not queries:
        raise ValueError("At least five documents and one query are required")
    if len({row["id"] for row in queries}) != len(queries):
        raise ValueError("Query IDs must be unique")
    for row in queries:
        if row["category"] not in {"exact_lexical", "semantic", "mixed"}:
            raise ValueError(f"Invalid category: {row['category']}")
        if not row["query"].strip() or row["expected_source"] not in sources:
            raise ValueError(f"Invalid query or expected source: {row['id']}")

    # Capture the inputs before loading models or observing any retrieval results.
    dataset_hashes = fingerprint(files + [queries_path])
    manifest_hashes = {}
    if args.dataset == "challenge":
        # This manifest was retained before the first challenge retrieval run.
        frozen_hashes = {}
        for line in CHALLENGE_MANIFEST.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            frozen_hashes[name] = digest
        if dataset_hashes != frozen_hashes:
            raise ValueError("Challenge inputs differ from the pre-retrieval freeze manifest")
        manifest_hashes = fingerprint([CHALLENGE_MANIFEST])
    sys.path.insert(0, str(ROOT / "src"))
    import config
    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if config.TOP_K != 5:
        raise ValueError("This baseline requires the existing TOP_K=5")

    # Override only data locations, before modules import configuration constants.
    runtime = ROOT / "bench/.runtime" / args.dataset
    runtime.mkdir(parents=True, exist_ok=True)
    config.DOCUMENTS_DIR = str(corpus)
    config.FAISS_INDEX_PATH = str(runtime / "index.faiss")
    config.RAG_DB_PATH = str(runtime / "rag.db")

    from rag import build_index, chunk, ingest

    # Stable document order; all parsing, chunking, embedding and indexing remain
    # the production implementations. Never load documents outside this corpus.
    documents = sorted(ingest.ingest_documents(), key=lambda doc: doc["path"])
    if {doc["path"] for doc in documents} != {str(path) for path in files}:
        raise ValueError("Ingestion did not load the complete evaluation corpus")
    token_counts = {
        Path(doc["path"]).relative_to(ROOT).as_posix(): len(chunk.encoder.encode(doc["text"]))
        for doc in documents
    }
    if any(len(chunk.chunk_text(doc["text"])) != 1 for doc in documents):
        raise ValueError("Each evaluation document must produce exactly one chunk")
    with patch.object(build_index, "ingest_documents", return_value=documents):
        build_index.build_index()

    from rag import query

    if not query._ensure_index_exists():
        raise ValueError("The rebuilt vector index could not be loaded")

    def source_name(source):
        return Path(source).resolve().relative_to(ROOT).as_posix()

    indexed_sources = [source_name(item["source"]) for item in query.chunks]
    if Counter(indexed_sources) != Counter(sources) or query.index.ntotal != len(files):
        raise ValueError("The rebuilt index must contain one chunk per document")

    warmup_query = "internal knowledge base"
    query.retrieve_vector(warmup_query)
    rows = []
    for record in queries:
        started = perf_counter()
        contexts = query.retrieve_vector(record["query"])
        latency_ms = (perf_counter() - started) * 1000
        ranked_sources = [source_name(context["source"]) for context in contexts]
        rank = next((i for i, source in enumerate(ranked_sources, 1)
                     if source == record["expected_source"]), None)
        rows.append({
            **record,
            "ranked_sources": ranked_sources,
            "expected_source_rank": rank,
            "hit_at_5": int(rank is not None),
            "reciprocal_rank": 1 / rank if rank is not None else 0.0,
            "latency_ms": latency_ms,
        })

    if fingerprint(sorted(corpus.glob("*.txt")) + [queries_path]) != dataset_hashes:
        raise ValueError("Evaluation inputs changed during measurement")
    if fingerprint([ROOT / path for path in manifest_hashes]) != manifest_hashes:
        raise ValueError("Freeze manifest changed during measurement")
    result = {
        "metadata": {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "retrieval_mode": "vector_only",
            "dataset": args.dataset,
            "corpus_path": corpus_name,
            "queries_path": queries_name,
            "freeze_manifest_sha256": manifest_hashes,
            "retrieval_entrypoint": "rag.query.retrieve_vector",
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "implementation_sha256": fingerprint([
                ROOT / "src/config.py", *sorted((ROOT / "src/rag").glob("*.py")),
                Path(__file__).resolve(),
            ]),
            "dataset_sha256": hashlib.sha256(
                json.dumps(dataset_hashes, sort_keys=True).encode()
            ).hexdigest(),
            "input_sha256": dataset_hashes,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_model_revision": query.model[0].auto_model.config._commit_hash,
            "embedding_device": str(query.model.device),
            "top_k": config.TOP_K,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "tokenizer": chunk.encoder.name,
            "document_count": len(files),
            "chunk_count": len(query.chunks),
            "query_count": len(queries),
            "document_token_counts": token_counts,
            "index_source_order": indexed_sources,
            "faiss_index_type": type(query.index).__name__,
            "faiss_metric_type": query.index.metric_type,
            "embedding_dimension": query.index.d,
            "vector_normalization": "L2 for documents and queries (production behavior)",
            "seed": 0,
            "torch_threads": torch.get_num_threads(),
            "faiss_threads": query.faiss.omp_get_max_threads(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {name: version(name) for name in (
                "sentence-transformers", "transformers", "torch", "faiss-cpu",
                "numpy", "tiktoken", "huggingface-hub", "tokenizers",
            )},
            "latency_protocol": {
                "clock": "time.perf_counter",
                "warmup_query": warmup_query,
                "warmup_calls": 1,
                "timed_calls_per_query": 1,
                "includes": "query embedding, normalization, FAISS search, result assembly",
                "excludes": "index build/load, model load, warmup, scoring, answer generation",
                "p95_method": "nearest rank",
            },
            "metric_definition": "Hit@5: expected source in returned top 5. MRR@5: mean "
                                 "1/rank of expected source, zero when absent. One relevant "
                                 "source and one chunk per document; no source deduplication.",
        },
        "queries": rows,
        "aggregate": {
            "all": aggregate(rows),
            "by_category": {
                category: aggregate([row for row in rows if row["category"] == category])
                for category in sorted({row["category"] for row in rows})
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
