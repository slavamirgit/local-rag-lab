from contextlib import chdir, closing, ExitStack, redirect_stdout
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
        # Keep tokenizer initialization out of these isolated builder tests.
        chunking = ModuleType("rag.chunk")
        chunking.chunk_documents = Mock()
        with patch.dict(sys.modules, {"rag.embed": embedding, "rag.chunk": chunking}):
            spec = importlib.util.spec_from_file_location(
                "build_index_under_test",
                Path(__file__).resolve().parents[1] / "src/rag/build_index.py",
            )
            self.builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.builder)

        from rag import fts, store

        self.fts = fts
        self.store = store
        self.legacy_path = self.directory / "fts_index.db"
        self.legacy_bytes = b"legacy sentinel: must never be read or replaced"
        self.rag_path = self.directory / "rag.db"
        stack.enter_context(patch.object(fts, "RAG_DB_PATH", str(self.rag_path)))
        stack.enter_context(patch.object(self.builder, "RAG_DB_PATH", str(self.rag_path)))
        stack.enter_context(patch.object(self.builder, "FAISS_INDEX_PATH", str(self.directory / "index.faiss")))
        stack.enter_context(patch.object(self.builder, "CHUNKS_PATH", str(self.directory / "chunks.pkl")))
        self.documents = [{"path": "z.txt", "text": "example"}]
        self.chunks = [
            {"text": "newneedle zirconium", "source": "z.txt", "chunk_id": 0},
            {"text": "hafnium", "source": "a.txt", "chunk_id": 0},
            {"text": "tantalum", "source": "z.txt", "chunk_id": 1},
        ]
        self.ingest = stack.enter_context(patch.object(self.builder, "ingest_documents", return_value=self.documents))
        self.chunk = stack.enter_context(patch.object(self.builder, "chunk_documents", return_value=self.chunks))
        self.embed = embedding.embed_chunks
        self.build_rag = stack.enter_context(patch.object(self.builder, "build_rag_db", wraps=store.build_rag_db))

    def seed_old_generation(self):
        self.old_chunks = [
            {"text": "old background", "source": "old.txt", "chunk_id": 7},
            {"text": "oldneedle", "source": "old.txt", "chunk_id": 8},
        ]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[-1, 0], [0, -1]], dtype="float32"))
        faiss.write_index(index, str(self.directory / "index.faiss"))
        with (self.directory / "chunks.pkl").open("wb") as file:
            pickle.dump(self.old_chunks, file)
        self.legacy_path.write_bytes(self.legacy_bytes)
        self.store.build_rag_db(self.old_chunks, self.rag_path)
        self.assert_rag_contents(self.rag_path, self.old_chunks)
        self.assertEqual(self.fts.search_fts("oldneedle"), [1])
        self.assertEqual(pickle.loads((self.directory / "chunks.pkl").read_bytes()), self.old_chunks)
        return self.artifact_bytes()

    def artifact_bytes(self):
        return {
            name: (self.directory / name).read_bytes()
            for name in ("index.faiss", "chunks.pkl", "rag.db")
        }

    def assert_rag_contents(self, path, chunks):
        self.assertEqual(self.store.load_chunks(path), chunks)
        with closing(sqlite3.connect(path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT id FROM chunks ORDER BY id"
            ).fetchall(), [(position,) for position in range(len(chunks))])
            for position, chunk in enumerate(chunks):
                self.assertEqual(connection.execute(
                    "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?",
                    (chunk["text"],),
                ).fetchall(), [(position,)])

    def assert_old_generation_unchanged(self, before):
        self.assertEqual(self.artifact_bytes(), before)
        self.assert_rag_contents(self.rag_path, self.old_chunks)
        chunks = pickle.loads((self.directory / "chunks.pkl").read_bytes())
        self.assertEqual(chunks, self.old_chunks)
        self.assertEqual(self.fts.search_fts("oldneedle"), [1])
        self.assertEqual(chunks[1], self.old_chunks[1])
        self.assertEqual(self.fts.search_fts("newneedle"), [])
        self.assertEqual(self.legacy_path.read_bytes(), self.legacy_bytes)
        self.assertEqual({path.name for path in self.directory.iterdir()},
                         set(before) | {self.legacy_path.name})

    def test_success_shares_ordered_chunks_and_persists_all_artifacts(self):
        self.seed_old_generation()
        replace = self.builder.os.replace

        def publish(source, destination):
            # Even the first replacement must wait for all three completed files.
            if replacements.call_count == 1:
                self.build_rag.assert_called_once()
                staged_index, = self.directory.glob(".index.faiss.*.tmp")
                staged_chunks, = self.directory.glob(".chunks.pkl.*.tmp")
                staged_rag = self.build_rag.call_args.args[1]
                self.assertEqual(
                    set(self.directory.glob(".*.tmp")),
                    {staged_index, staged_chunks, staged_rag},
                )
                self.assertEqual(faiss.read_index(str(staged_index)).ntotal, len(self.chunks))
                self.assertEqual(pickle.loads(staged_chunks.read_bytes()), self.chunks)
                self.assertEqual(self.fts.search_fts("newneedle", db_path=staged_rag), [0])
                self.assert_rag_contents(staged_rag, self.chunks)
            self.assertEqual(Path(source).parent, Path(destination).parent)
            replace(source, destination)

        with (
            patch.object(self.builder.os, "replace", side_effect=publish) as replacements,
            patch.object(self.builder.pickle, "dump", wraps=pickle.dump) as dump,
        ):
            self.builder.build_index()
        self.assertEqual(replacements.call_count, 3)
        self.assertEqual(
            [Path(call.args[1]) for call in replacements.call_args_list],
            [self.directory / "index.faiss", self.directory / "chunks.pkl", self.rag_path],
        )
        dump.assert_called_once()
        self.assertIs(dump.call_args.args[0], self.chunks)

        self.ingest.assert_called_once_with()
        self.chunk.assert_called_once_with(self.documents)
        self.embed.assert_called_once_with(self.chunks)
        self.assertIs(self.embed.call_args.args[0], self.chunks)
        self.build_rag.assert_called_once()
        self.assertIs(self.build_rag.call_args.args[0], self.chunks)
        staged_rag = self.build_rag.call_args.args[1]
        self.assertEqual(staged_rag.parent, self.rag_path.parent)
        self.assertNotEqual(staged_rag, self.rag_path)

        with (self.directory / "chunks.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), self.chunks)
        index = faiss.read_index(str(self.directory / "index.faiss"))
        self.assertIsInstance(index, faiss.IndexFlatIP)
        self.assertEqual(index.ntotal, len(self.chunks))
        np.testing.assert_allclose(index.reconstruct_n(0, 3), [[1, 0], [0, 1], [0.6, 0.8]])

        with closing(sqlite3.connect(self.rag_path)) as connection:
            self.assertEqual(connection.execute(
                "SELECT rowid, text FROM chunks_fts ORDER BY rowid"
            ).fetchall(), list(enumerate(chunk["text"] for chunk in self.chunks)))
        for position, chunk in enumerate(self.chunks):
            self.assertEqual(self.fts.search_fts(chunk["text"]), [position])
        self.assertEqual(self.fts.search_fts("oldneedle"), [])
        self.assert_rag_contents(self.rag_path, self.chunks)
        self.assertEqual(self.legacy_path.read_bytes(), self.legacy_bytes)
        self.assertEqual(
            {path.name for path in self.directory.iterdir()},
            {"index.faiss", "chunks.pkl", "fts_index.db", "rag.db"},
        )

    def test_rag_build_failure_preserves_old_generation_and_cleans_staging(self):
        before = self.seed_old_generation()
        error = sqlite3.OperationalError("rag.db build failed")

        def fail_rag(chunks, db_path):
            self.assertIs(chunks, self.chunks)
            self.assertNotEqual(db_path, self.rag_path)
            self.assertEqual(db_path.parent, self.rag_path.parent)
            staged_index, = self.directory.glob(".index.faiss.*.tmp")
            staged_chunks, = self.directory.glob(".chunks.pkl.*.tmp")
            self.assertEqual(faiss.read_index(str(staged_index)).ntotal, len(self.chunks))
            self.assertEqual(pickle.loads(staged_chunks.read_bytes()), self.chunks)
            self.assertEqual(self.artifact_bytes(), before)
            # Fail inside the real SQLite transaction, after inserting chunks.
            with patch.object(self.store, "_rebuild_fts", side_effect=error):
                try:
                    self.store.build_rag_db(chunks, db_path)
                finally:
                    for suffix in ("-journal", "-wal", "-shm"):
                        Path(str(db_path) + suffix).write_bytes(b"partial SQLite data")

        self.build_rag.side_effect = fail_rag
        with (
            patch.object(self.builder.os, "replace") as replacements,
            self.assertRaises(sqlite3.OperationalError) as raised,
        ):
            self.builder.build_index()
        self.assertIs(raised.exception, error)
        replacements.assert_not_called()
        self.build_rag.assert_called_once()
        self.assert_old_generation_unchanged(before)
        self.assertNotIn("Indexing complete", self.output.getvalue())

    def test_earlier_construction_failures_preserve_old_generation(self):
        before = self.seed_old_generation()
        error = OSError("injected construction failure")

        def fail_faiss(index, path):
            Path(path).write_bytes(b"partial FAISS")
            raise error

        def fail_pickle(chunks, file):
            file.write(b"partial pickle")
            raise error

        failures = [
            (self.builder, "embed_chunks", error),
            (self.builder.faiss, "IndexFlatIP", error),
            (self.builder.faiss, "normalize_L2", error),
            (self.builder.faiss, "write_index", fail_faiss),
            (self.builder.pickle, "dump", fail_pickle),
        ]
        for target, name, failure in failures:
            with (
                self.subTest(stage=name),
                patch.object(target, name, side_effect=failure),
                patch.object(self.builder.os, "replace") as replacements,
            ):
                with self.assertRaises(OSError) as raised:
                    self.builder.build_index()
                self.assertIs(raised.exception, error)
                replacements.assert_not_called()
            self.assert_old_generation_unchanged(before)
        self.build_rag.assert_not_called()

    def test_rag_failure_on_first_build_publishes_nothing(self):
        error = sqlite3.OperationalError("first rag.db build failed")
        with (
            patch.object(self.store, "_rebuild_fts", side_effect=error),
            patch.object(self.builder.os, "replace") as replacements,
            self.assertRaises(sqlite3.OperationalError) as raised,
        ):
            self.builder.build_index()
        self.assertIs(raised.exception, error)
        self.build_rag.assert_called_once()
        replacements.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_relative_paths_keep_existing_base_directories(self):
        src_dir = self.directory / "src"
        src_dir.mkdir()
        cwd = self.directory / "working"
        cwd.mkdir()
        with (
            chdir(cwd),
            patch.object(self.builder, "__file__", str(src_dir / "rag/build_index.py")),
            patch.object(self.builder, "FAISS_INDEX_PATH", "index.faiss"),
            patch.object(self.builder, "CHUNKS_PATH", "chunks.pkl"),
            patch.object(self.builder, "RAG_DB_PATH", "storage/rag.db"),
        ):
            self.builder.build_index()
        self.assertEqual(
            {str(path.relative_to(self.directory)) for path in self.directory.rglob("*") if path.is_file()},
            {"src/index.faiss", "src/chunks.pkl", "working/storage/rag.db"},
        )
        self.assertEqual(
            self.fts.search_fts("newneedle", db_path=cwd / "storage/rag.db"), [0]
        )
        self.assert_rag_contents(cwd / "storage/rag.db", self.chunks)

    def test_no_documents_retains_existing_early_return(self):
        self.ingest.return_value = []
        self.builder.build_index()
        self.chunk.assert_not_called()
        self.embed.assert_not_called()
        self.build_rag.assert_not_called()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_no_documents_preserves_old_generation(self):
        before = self.seed_old_generation()
        self.ingest.return_value = []
        self.builder.build_index()
        self.chunk.assert_not_called()
        self.embed.assert_not_called()
        self.build_rag.assert_not_called()
        self.assert_old_generation_unchanged(before)


if __name__ == "__main__":
    unittest.main()
