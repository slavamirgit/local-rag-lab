"""Optional lexical search terms; independent of retrieval and answer generation."""

import json
import logging
import re

import requests

try:  # Support both src.rag imports and running with src on PYTHONPATH.
    from ..config import (
        OLLAMA_MODEL,
        OLLAMA_URL,
        QUERY_EXPANSION_MAX_TERMS,
        QUERY_EXPANSION_TEMPERATURE,
        QUERY_EXPANSION_TIMEOUT,
    )
except ImportError:
    from config import (
        OLLAMA_MODEL,
        OLLAMA_URL,
        QUERY_EXPANSION_MAX_TERMS,
        QUERY_EXPANSION_TEMPERATURE,
        QUERY_EXPANSION_TIMEOUT,
    )

logger = logging.getLogger(__name__)


def _parse_terms(output: str, query: str) -> list[str]:
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    if output.startswith("```") and output.endswith("```"):
        output = re.sub(r"^```(?:json)?\s*", "", output)[:-3].strip()
    try:
        values = json.loads(output)
    except (ValueError, RecursionError):
        # Do not turn broken JSON or incomplete thinking into lexical terms.
        if output.startswith(("[", "{", "<", "```")) or "," not in output:
            return []
        values = output.split(",")
    if isinstance(values, dict):
        values = values.get("terms", values.get("keywords", []))
    if not isinstance(values, list):
        return []

    terms = []
    # Case-sensitive deduplication preserves distinct technical identifiers.
    seen = {query.strip()}
    for value in values:
        if not isinstance(value, str):
            continue
        term = value.strip().strip(" \t\r\n\"'`,;")
        if not term or not any(char.isalnum() for char in term) or term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms[:QUERY_EXPANSION_MAX_TERMS]


def generate_search_terms(query: str) -> list[str]:
    """Return at most MAX_TERMS additional terms, or [] on model failure."""
    if not query.strip():
        return []

    prompt = (
        f"Extract up to {QUERY_EXPANSION_MAX_TERMS} keywords or short phrases "
        "from the question for lexical document search. Use only terms from the question. "
        "Keep exact filenames, file paths, technical identifiers, and CLI commands "
        "or meaningful command fragments intact. Do not split src/config.py into "
        "src and config.py. Avoid generic path components or category words when "
        "a more specific exact term is available. Preserve spelling; use fewer terms if sufficient. "
        "Do not answer the question or repeat it in full. "
        "Return only a JSON array of strings, no explanatory prose.\n"
        f"Question: {json.dumps(query)}"
    )
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "format": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": QUERY_EXPANSION_MAX_TERMS,
                },
                "options": {
                    "temperature": QUERY_EXPANSION_TEMPERATURE,
                    "num_predict": 128,
                },
            },
            timeout=QUERY_EXPANSION_TIMEOUT,
        )
        response.raise_for_status()
        body = response.json()
        output = body.get("response") if isinstance(body, dict) else None
        if not isinstance(output, str):
            logger.warning("Query expansion returned an unexpected response")
            return []
        terms = _parse_terms(output, query)
        if not terms:
            logger.warning("Query expansion returned no usable additional terms")
        return terms
    except (requests.RequestException, ValueError, RecursionError) as exc:
        logger.warning("Query expansion failed: %s", exc)
        return []


def expand_query(query: str) -> list[str]:
    """Preserve the exact non-blank query first, even if generation fails."""
    if not query.strip():
        return []
    return [query, *generate_search_terms(query)]
