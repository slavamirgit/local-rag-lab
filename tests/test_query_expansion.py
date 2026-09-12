import json
import unittest
from unittest.mock import patch

import requests

from src.rag import query_expansion as expansion


class QueryExpansionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(expansion.requests, "post")
        self.post = patcher.start()
        self.addCleanup(patcher.stop)
        self.response = self.post.return_value

    def output(self, text):
        self.response.json.return_value = {"response": text}

    def test_valid_array_and_deterministic_bounded_request(self):
        self.output('["OLLAMA_MODEL", "config.py"]')
        self.assertEqual(
            expansion.generate_search_terms("Where is the model configured?"),
            ["OLLAMA_MODEL", "config.py"],
        )
        self.response.raise_for_status.assert_called_once_with()
        self.post.assert_called_once()
        args, kwargs = self.post.call_args
        self.assertEqual(args, (expansion.OLLAMA_URL,))
        self.assertEqual(kwargs["timeout"], expansion.QUERY_EXPANSION_TIMEOUT)
        self.assertGreater(kwargs["timeout"], 0)
        payload = kwargs["json"]
        self.assertEqual(payload["model"], expansion.OLLAMA_MODEL)
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertEqual(payload["format"]["type"], "array")
        self.assertEqual(payload["format"]["items"], {"type": "string"})
        self.assertEqual(payload["format"]["maxItems"], expansion.QUERY_EXPANSION_MAX_TERMS)
        self.assertIn("no explanatory prose", payload["prompt"])
        self.assertIn("Do not answer", payload["prompt"])
        self.assertIn(
            "Do not split src/config.py into src and config.py.",
            payload["prompt"],
        )

    def test_limit_applies_after_normalization(self):
        terms = [f"term_{i}" for i in range(expansion.QUERY_EXPANSION_MAX_TERMS + 3)]
        self.output(json.dumps(["", None, "query", terms[0], terms[0], *terms]))
        self.assertEqual(
            expansion.generate_search_terms("query"),
            terms[:expansion.QUERY_EXPANSION_MAX_TERMS],
        )

    def test_deduplicates_and_discards_junk_without_reordering(self):
        self.output(json.dumps([
            " query ", " `config.py` ", "config.py", None, 42, False,
            {}, [], "", "  ", "...", " ; ", "'OLLAMA_MODEL',",
        ]))
        self.assertEqual(expansion.generate_search_terms("query"), ["config.py", "OLLAMA_MODEL"])

    def test_preserves_identifiers_filenames_commands_and_versions(self):
        terms = ["OLLAMA_MODEL", ".env", "src/config.py", "git diff --check", "qwen3:0.6b"]
        self.output(json.dumps(terms))
        self.assertEqual(expansion.generate_search_terms("query"), terms)

    def test_preserves_case_sensitive_identifiers(self):
        self.output('["Foo_Bar", "foo_bar", "--dry-run", "v1.2.3"]')
        self.assertEqual(
            expansion.generate_search_terms("query"),
            ["Foo_Bar", "foo_bar", "--dry-run", "v1.2.3"],
        )

    def test_json_object(self):
        for field in ("terms", "keywords"):
            with self.subTest(field=field):
                self.output(json.dumps({field: ["config.py", "OLLAMA_MODEL"]}))
                self.assertEqual(expansion.generate_search_terms("query"), ["config.py", "OLLAMA_MODEL"])

    def test_comma_separated_fallback(self):
        self.output(' config.py, "OLLAMA_MODEL", `git diff --check`, , config.py ')
        self.assertEqual(
            expansion.generate_search_terms("query"),
            ["config.py", "OLLAMA_MODEL", "git diff --check"],
        )

    def test_code_fences_and_thinking_wrappers(self):
        for output in (
            '```json\n["config.py"]\n```',
            '```\n["config.py"]\n```',
            '<think>internal, reasoning\ntext</think>\n["config.py"]',
            '<think>internal reasoning</think>\n```json\n["config.py"]\n```',
        ):
            with self.subTest(output=output):
                self.output(output)
                self.assertEqual(expansion.generate_search_terms("query"), ["config.py"])

    def test_malformed_empty_or_unusable_output(self):
        for output in (
            '["config.py", broken]', '{"terms": ["config.py",}',
            "not valid output", "", " \n ", "[]", "null", '"config.py"',
            '[null, 7, false, {}, [], "", "..."]', '["query"]',
            '{"answer": "config.py"}', '{"terms": 42}',
            '{"keywords": "config.py"}', '<think>unfinished, reasoning',
        ):
            with self.subTest(output=output):
                self.output(output)
                self.assertEqual(expansion.generate_search_terms("query"), [])
                self.assertEqual(expansion.expand_query("query"), ["query"])

    def test_unexpected_http_response_structure(self):
        for body in ({}, [], None, 12, "text", {"response": None}, {"response": []}, {"response": {}}):
            with self.subTest(body=body):
                self.response.json.return_value = body
                self.assertEqual(expansion.generate_search_terms("query"), [])
                self.assertEqual(expansion.expand_query("query"), ["query"])

    def test_malformed_http_json(self):
        self.response.json.side_effect = ValueError("invalid JSON")
        self.assertEqual(expansion.generate_search_terms("query"), [])
        self.assertEqual(expansion.expand_query("query"), ["query"])

    def test_http_failure(self):
        self.response.raise_for_status.side_effect = requests.HTTPError("503")
        self.assertEqual(expansion.generate_search_terms("query"), [])
        self.assertEqual(expansion.expand_query("query"), ["query"])
        self.response.json.assert_not_called()

    def test_connection_request_and_timeout_failures(self):
        for error in (requests.ConnectionError, requests.RequestException, requests.Timeout):
            with self.subTest(error=error):
                self.post.side_effect = error("unavailable")
                self.assertEqual(expansion.generate_search_terms("query"), [])
                self.assertEqual(expansion.expand_query("query"), ["query"])

    def test_expand_preserves_exact_original_first(self):
        self.output('["config.py", "query", "config.py", "OLLAMA_MODEL"]')
        self.assertEqual(
            expansion.expand_query("  query  "),
            ["  query  ", "config.py", "OLLAMA_MODEL"],
        )

    def test_blank_queries_do_not_request_ollama(self):
        for query in ("", " ", "\t\n"):
            with self.subTest(query=query):
                self.assertEqual(expansion.generate_search_terms(query), [])
                self.assertEqual(expansion.expand_query(query), [])
        self.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
