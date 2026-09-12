from contextlib import closing, ExitStack, redirect_stdout
import importlib.util
import io
from pathlib import Path
import pickle
import sqlite3
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import faiss
import numpy as np


class BuildIndexTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.directory = Path(stack.enter_context(TemporaryDirectory()))
        self.output = stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(sys, "path", sys.path.copy()))

        # Replace the embedding boundary before importing the builder: embed.py
        # otherwise constructs a SentenceTransformer at module import time.
        embedding = ModuleType("rag.embed")
        embedding.embed_chunks = Mock(return_value=np.array(
            [[3, 0], [0, 4], [3, 4]], dtype="float32"
        ))
        with patch.dict(sys.modules, {"rag.embed": embedding}):
            spec = importlib.util.spec_from_file_location(
                "build_index_under_test",
                Path(__file__).resolve().parents[1] / "src/rag/build_index.py",
            )
            self.builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.builder)

        from rag import fts

        self.fts = fts
        self.fts_path = self.directory / "fts_index.db"
        stack.enter_context(patch.object(fts, "FTS_INDEX_PATH", str(self.fts_path)))
        stack.enter_context(patch.object(self.builder, "FAISS_INDEX_PATH", str(self.directory / "index.faiss")))
        stack.enter_context(patch.object(self.builder, "CHUNKS_PATH", str(self.directory / "chunks.pkl")))
        self.documents = [{"path": "z.txt", "text": "example"}]
        self.chunks = [
            {"text": "zirconium", "source": "z.txt", "chunk_id": 0},
            {"text": "hafnium", "source": "a.txt", "chunk_id": 0},
            {"text": "tantalum", "source": "z.txt", "chunk_id": 1},
        ]
        self.ingest = stack.enter_context(patch.object(self.builder, "ingest_documents", return_value=self.documents))
        self.chunk = stack.enter_context(patch.object(self.builder, "chunk_documents", return_value=self.chunks))
        self.embed = embedding.embed_chunks
        self.build_fts = stack.enter_context(patch.object(self.builder, "build_fts_index", wraps=fts.build_fts_index))

    def test_success_shares_ordered_chunks_and_persists_all_artifacts(self):
        self.builder.build_index()

        self.ingest.assert_called_once_with()
        self.chunk.assert_called_once_with(self.documents)
        self.embed.assert_called_once_with(self.chunks)
        self.assertIs(self.embed.call_args.args[0], self.chunks)
        self.build_fts.assert_called_once_with(self.chunks)
        self.assertIs(self.build_fts.call_args.args[0], self.chunks)

        with (self.directory / "chunks.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), self.chunks)
        index = faiss.read_index(str(self.directory / "index.faiss"))
        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertEqual(index.ntotal, len(self.chunks))
        np.testing.assert_allclose(index.reconstruct_n(0, 3), [[1, 0], [0, 1], [0.6, 0.8]])

        with closing(sqlite3.connect(self.fts_path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT rowid - 1, text FROM chunks_fts ORDER BY rowid"
            ).fetchall(), list(enumerate(chunk["text"] for chunk in self.chunks)))
        for position, chunk in enumerate(self.chunks):
            self.assertEqual(self.fts.search_fts(chunk["text"]), [position])

    def test_fts_build_exception_propagates(self):
        error = sqlite3.OperationalError("FTS build failed")
        self.build_fts.side_effect = error
        with self.assertRaises(sqlite3.OperationalError) as raised:
            self.builder.build_index()
        self.assertIs(raised.exception, error)
        self.build_fts.assert_called_once_with(self.chunks)
        self.assertNotIn("Indexing complete", self.output.getvalue())

    def test_no_documents_retains_existing_early_return(self):
        self.ingest.return_value = []
        self.builder.build_index()
        self.chunk.assert_not_called()
        self.embed.assert_not_called()
        self.build_fts.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
