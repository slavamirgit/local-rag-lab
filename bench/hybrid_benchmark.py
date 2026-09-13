"""Measure production hybrid retrieval against frozen vector-only datasets."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

from vector_baseline import (
    CHALLENGE_MANIFEST, DATASETS, ROOT, aggregate, fingerprint, validate_output_path,
)


def verify_query_identity(vector_rows, hybrid_rows):
    """Require identical ordered query identities and labels before comparison."""
    fields = ("id", "category", "query", "expected_source")
    if len({row["id"] for row in hybrid_rows}) != len(hybrid_rows):
        raise ValueError("Query IDs must be unique")
    if [[row[key] for key in fields] for row in vector_rows] != [
        [row[key] for key in fields] for row in hybrid_rows
    ]:
        raise ValueError("Queries or labels differ from the frozen vector measurement")


def verify_inputs(dataset, baseline):
    corpus_name, queries_name, _ = DATASETS[dataset]
    hashes = fingerprint(sorted((ROOT / corpus_name).glob("*.txt")) + [ROOT / queries_name])
    if hashes != baseline["metadata"]["input_sha256"]:
        raise ValueError("Corpus/query hashes differ from the frozen vector measurement")
    manifest_hashes = {}
    if dataset == "challenge":
        frozen = {}
        for line in CHALLENGE_MANIFEST.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            frozen[name] = digest
        if hashes != frozen:
            raise ValueError("Challenge inputs differ from the freeze manifest")
        manifest_hashes = fingerprint([CHALLENGE_MANIFEST])
        if manifest_hashes != baseline["metadata"]["freeze_manifest_sha256"]:
            raise ValueError("Challenge freeze manifest changed")
    return hashes, manifest_hashes


def measure_query(query_module, text):
    """Observe the one real expansion call, leaving production fallback intact."""
    real_expand = query_module.expand_query
    observation = {"expansion_raised_unexpectedly": False}
    calls = 0

    def capture(original):
        nonlocal calls
        calls += 1
        try:
            inputs = real_expand(original)
        except Exception as exc:
            observation["expansion_raised_unexpectedly"] = True
            observation["expansion_exception_type"] = type(exc).__name__
            raise
        observation["expansion_inputs"] = inputs
        return inputs

    with patch.object(query_module, "expand_query", capture):
        started = perf_counter()
        contexts = query_module.retrieve(text)
        latency_ms = (perf_counter() - started) * 1000
    if calls != 1:
        raise ValueError("Expected exactly one expansion call per retrieve()")
    if observation["expansion_raised_unexpectedly"]:
        observation["expansion_inputs"] = [text]
    return contexts, latency_ms, observation


def metric_comparison(vector, hybrid):
    result = {}
    for key in ("hit_at_5", "mrr_at_5"):
        result[f"vector_{key}"] = vector[key]
        result[f"hybrid_{key}"] = hybrid[key]
        result[f"{key}_delta"] = hybrid[key] - vector[key]
    result["vector_mean_latency_ms"] = vector["latency_ms"]["mean"]
    result["hybrid_mean_latency_ms"] = hybrid["latency_ms"]["mean"]
    return result


def compare_to_vector(baseline, rows, metrics):
    verify_query_identity(baseline["queries"], rows)
    comparisons = []
    for vector, hybrid in zip(baseline["queries"], rows):
        old, new = vector["expected_source_rank"], hybrid["expected_source_rank"]
        old_order = float("inf") if old is None else old
        new_order = float("inf") if new is None else new
        comparisons.append({
            "id": hybrid["id"],
            "vector_expected_source_rank": old,
            "hybrid_expected_source_rank": new,
            "rank_outcome": ("improved" if new_order < old_order else
                             "worsened" if new_order > old_order else "unchanged"),
            "recovered_miss": old is None and new is not None,
            "new_miss": old is not None and new is None,
        })
    return {
        "all": metric_comparison(baseline["aggregate"]["all"], metrics["all"]),
        "by_category": {
            category: metric_comparison(baseline["aggregate"]["by_category"][category], values)
            for category, values in metrics["by_category"].items()
        },
        "queries": comparisons,
        "rank_protocol": "A missing TOP_K source is worse than every visible rank; lower is better.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="sanity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = validate_output_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use --output with a new path")
    corpus_name, queries_name, vector_name = DATASETS[args.dataset]
    baseline_path = ROOT / "bench/results" / vector_name
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_hash = fingerprint([baseline_path])
    input_hashes, manifest_hashes = verify_inputs(args.dataset, baseline)
    records = json.loads((ROOT / queries_name).read_text(encoding="utf-8"))
    verify_query_identity(baseline["queries"], records)
    implementation_paths = [ROOT / "src/config.py", *sorted((ROOT / "src/rag").glob("*.py"))]
    implementation_hashes = fingerprint(implementation_paths)
    runner_hashes = fingerprint([Path(__file__).resolve(), ROOT / "bench/vector_baseline.py"])

    sys.path.insert(0, str(ROOT / "src"))
    import config
    import requests

    # Read-only availability check, not another expansion/generation request.
    # A sandbox/network failure aborts before measurement or writing a result.
    endpoint = urlsplit(config.OLLAMA_URL)
    tags_url = urlunsplit((endpoint.scheme, endpoint.netloc, "/api/tags", "", ""))
    response = requests.get(tags_url, timeout=config.QUERY_EXPANSION_TIMEOUT)
    response.raise_for_status()
    installed = response.json()["models"]
    model = next((item for item in installed
                  if config.OLLAMA_MODEL in (item.get("name"), item.get("model"))), None)
    if model is None:
        raise ValueError(f"Configured Ollama model is not installed: {config.OLLAMA_MODEL}")

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if config.TOP_K != 5:
        raise ValueError("Frozen vector comparison requires TOP_K=5")

    runtime = ROOT / "bench/.runtime" / args.dataset
    runtime.mkdir(parents=True, exist_ok=True)
    config.DOCUMENTS_DIR = str(ROOT / corpus_name)
    config.FAISS_INDEX_PATH = str(runtime / "index.faiss")
    config.RAG_DB_PATH = str(runtime / "rag.db")

    from rag import build_index, chunk, ingest

    files = sorted((ROOT / corpus_name).glob("*.txt"))
    documents = sorted(ingest.ingest_documents(), key=lambda doc: doc["path"])
    if {doc["path"] for doc in documents} != {str(path) for path in files}:
        raise ValueError("Ingestion did not load exactly the frozen corpus")
    with patch.object(build_index, "ingest_documents", return_value=documents):
        build_index.build_index()

    from rag import query

    if not query._ensure_index_exists():
        raise ValueError("The rebuilt vector index could not be loaded")

    def source_name(source):
        return Path(source).resolve().relative_to(ROOT).as_posix()

    indexed_sources = [source_name(item["source"]) for item in query.chunks]
    if (Counter(indexed_sources) != Counter(path.relative_to(ROOT).as_posix() for path in files)
            or query.index.ntotal != len(files)):
        raise ValueError("Expected one indexed chunk per frozen document")

    warmup_query = "internal knowledge base"
    _, _, warmup_observation = measure_query(query, warmup_query)
    rows = []
    for record in records:
        contexts, latency_ms, observation = measure_query(query, record["query"])
        ranked_sources = [source_name(context["source"]) for context in contexts]
        rank = next((i for i, source in enumerate(ranked_sources, 1)
                     if source == record["expected_source"]), None)
        rows.append({**record, **observation, "ranked_sources": ranked_sources,
                     "expected_source_rank": rank, "hit_at_5": int(rank is not None),
                     "reciprocal_rank": 1 / rank if rank is not None else 0.0,
                     "latency_ms": latency_ms})

    if verify_inputs(args.dataset, baseline) != (input_hashes, manifest_hashes):
        raise ValueError("Inputs changed during measurement")
    if (fingerprint([baseline_path]) != baseline_hash
            or fingerprint(implementation_paths) != implementation_hashes
            or fingerprint([ROOT / name for name in runner_hashes]) != runner_hashes):
        raise ValueError("Baseline or implementation changed during measurement")
    metrics = {
        "all": aggregate(rows),
        "by_category": {category: aggregate([row for row in rows if row["category"] == category])
                        for category in sorted({row["category"] for row in rows})},
    }
    result = {
        "metadata": {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "retrieval_mode": "hybrid",
            "retrieval_entrypoint": "rag.query.retrieve",
            "dataset": args.dataset,
            "corpus_path": corpus_name,
            "queries_path": queries_name,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "implementation_sha256": implementation_hashes,
            "benchmark_runner_sha256": runner_hashes,
            "input_sha256": input_hashes,
            "freeze_manifest_sha256": manifest_hashes,
            "vector_result_sha256": baseline_hash,
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_model_revision": query.model[0].auto_model.config._commit_hash,
            "embedding_device": str(query.model.device),
            "ollama_model": config.OLLAMA_MODEL,
            "ollama_model_digest": model.get("digest"),
            "top_k": config.TOP_K,
            "vector_candidates": config.VECTOR_CANDIDATES,
            "fts_candidates": config.FTS_CANDIDATES,
            "rrf_k": config.RRF_K,
            "rrf_weights": "production default equal weights",
            "query_expansion_max_terms": config.QUERY_EXPANSION_MAX_TERMS,
            "query_expansion_temperature": config.QUERY_EXPANSION_TEMPERATURE,
            "query_expansion_timeout": config.QUERY_EXPANSION_TIMEOUT,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "document_count": len(files),
            "chunk_count": len(query.chunks),
            "query_count": len(rows),
            "index_source_order": indexed_sources,
            "tokenizer": chunk.encoder.name,
            "faiss_index_type": type(query.index).__name__,
            "faiss_metric_type": query.index.metric_type,
            "embedding_dimension": query.index.d,
            "seed": 0,
            "torch_threads": torch.get_num_threads(),
            "faiss_threads": query.faiss.omp_get_max_threads(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": {name: version(name) for name in (
                "sentence-transformers", "transformers", "torch", "faiss-cpu", "numpy",
                "tiktoken", "huggingface-hub", "tokenizers", "requests")},
            "latency_protocol": {
                "clock": "time.perf_counter",
                "warmup_query": warmup_query,
                "warmup_calls": 1,
                "warmup_observation": warmup_observation,
                "timed_calls_per_query": 1,
                "includes": "real Query Expansion/Ollama call, parallel vector/FTS searches, "
                            "RRF, final chunk mapping, expansion observation overhead",
                "excludes": "availability check, model/index initial loading, index build, "
                            "warmup, scoring, comparison, serialization, final answer generation",
                "p95_method": "nearest rank",
            },
            "metric_definition": baseline["metadata"]["metric_definition"],
        },
        "queries": rows,
        "aggregate": metrics,
        "comparison_to_vector": compare_to_vector(baseline, rows, metrics),
    }
    serialized = json.dumps(result, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation retains the first successful measurement at every path.
    with output.open("x", encoding="utf-8") as file:
        file.write(serialized)
    print(json.dumps(metrics, indent=2))
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
