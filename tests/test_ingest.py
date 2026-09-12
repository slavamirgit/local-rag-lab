import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

from src.rag import ingest


class IngestDocumentsTests(unittest.TestCase):
    def test_discovers_nested_supported_files_and_ignores_other_entries(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested" / "deeper"
            nested.mkdir(parents=True)
            (root / "empty.txt").mkdir()
            supported = [nested / f"document{suffix}" for suffix in (
                ".txt", ".md", ".pdf", ".docx"
            )]
            for path in supported + [root / ".gitkeep", nested / "ignored.csv"]:
                path.touch()

            # Isolate discovery from PDF/DOCX parsing while using real traversal.
            with patch.object(ingest, "DOCUMENTS_DIR", root), patch.object(
                ingest, "load_document", side_effect=lambda path: path.name
            ) as loader:
                documents = ingest.ingest_documents()

            self.assertCountEqual(documents, [
                {"path": str(path), "text": path.name} for path in supported
            ])
            self.assertCountEqual(loader.call_args_list, [
                call(path) for path in supported
            ])

    def test_gitkeep_only_returns_no_documents(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gitkeep").touch()

            with patch.object(ingest, "DOCUMENTS_DIR", root), patch.object(
                ingest, "load_document"
            ) as loader:
                self.assertEqual(ingest.ingest_documents(), [])

            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
