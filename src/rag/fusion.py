"""Pure reciprocal rank fusion over zero-based chunk positions."""

from math import isfinite
from numbers import Integral, Real

try:  # Support both src.rag imports and running with src on PYTHONPATH.
    from ..config import RRF_K
except ImportError:
    from config import RRF_K


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def reciprocal_rank_fusion(rankings, limit=None, k=None, weights=None) -> list[int]:
    """Rank chunk IDs by summed weight / (k + one-based rank), then ID.

    Duplicates contribute at their first original rank only. Zero-weight lists
    add no candidates. Invalid k, weights, or IDs raise ValueError, even when
    limit is nonpositive. Inputs are not modified; None returns all candidates.
    """
    k = _finite_number(RRF_K if k is None else k, "k")
    if k <= 0:
        raise ValueError("k must be positive")
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)
    else:
        try:
            weights = list(weights)
        except TypeError as exc:
            raise ValueError("weights must contain one number per ranking") from exc
        if len(weights) != len(rankings):
            raise ValueError("weights must contain one number per ranking")
        weights = [_finite_number(weight, "weight") for weight in weights]
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative")

    scores = {}
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if isinstance(chunk_id, bool) or not isinstance(chunk_id, Integral) or chunk_id < 0:
                raise ValueError("chunk IDs must be non-negative integers, excluding bool")
            chunk_id = int(chunk_id)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if weight:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)

    if limit is not None and limit <= 0:
        return []
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
