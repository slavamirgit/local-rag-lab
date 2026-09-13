from contextlib import chdir, closing, ExitStack, redirect_stdout
import ast
import hashlib
import importlib.util
import io
from pathlib import Path
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
        self.legacy_paths = (self.directory / "chunks.pkl", self.directory / "fts_index.db")
        self.legacy_bytes = b"legacy sentinel: must never be read or replaced"
        self.rag_path = self.directory / "rag.db"
        stack.enter_context(patch.object(fts, "RAG_DB_PATH", str(self.rag_path)))
        stack.enter_context(patch.object(self.builder, "RAG_DB_PATH", str(self.rag_path)))
        stack.enter_context(patch.object(self.builder, "FAISS_INDEX_PATH", str(self.directory / "index.faiss")))
        stack.enter_context(patch.object(self.builder, "__file__", str(self.directory / "rag/build_index.py")))
        stack.enter_context(chdir(self.directory))
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
            {"text": "previous appendix", "source": "old.txt", "chunk_id": 9},
        ]
        index = faiss.IndexFlatIP(2)
        index.add(np.array([[-1, 0], [0, -1], [-0.6, -0.8]], dtype="float32"))
        faiss.write_index(index, str(self.directory / "index.faiss"))
        self.seed_legacy_files()
        self.old_metadata = {
            "faiss_sha256": hashlib.sha256(
                (self.directory / "index.faiss").read_bytes()
            ).hexdigest(),
            "chunk_count": len(self.old_chunks),
        }
        self.store.build_rag_db(
            self.old_chunks, self.rag_path,
            faiss_sha256=self.old_metadata["faiss_sha256"],
        )
        self.assertEqual(self.store.load_generation_metadata(self.rag_path), self.old_metadata)
        self.assert_rag_contents(self.rag_path, self.old_chunks)
        self.assertEqual(self.fts.search_fts("oldneedle"), [1])
        return self.artifact_bytes()

    def seed_legacy_files(self):
        for path in self.legacy_paths:
            path.write_bytes(self.legacy_bytes)

    def assert_legacy_unchanged(self):
        for path in self.legacy_paths:
            self.assertEqual(path.read_bytes(), self.legacy_bytes)

    def artifact_bytes(self):
        return {
            name: (self.directory / name).read_bytes()
            for name in ("index.faiss", "rag.db")
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
        self.assertEqual(self.store.load_generation_metadata(self.rag_path), self.old_metadata)
        self.assert_rag_contents(self.rag_path, self.old_chunks)
        self.assertEqual(self.fts.search_fts("oldneedle"), [1])
        self.assertEqual(self.fts.search_fts("newneedle"), [])
        self.assert_legacy_unchanged()
        self.assertEqual({path.name for path in self.directory.iterdir()},
                         set(before) | {path.name for path in self.legacy_paths})

    def test_success_shares_ordered_chunks_and_persists_all_artifacts(self):
        self.seed_old_generation()
        replace = self.builder.os.replace

        def publish(source, destination):
            self.assert_legacy_unchanged()
            # Even the first replacement must wait for both completed files.
            if replacements.call_count == 1:
                self.build_rag.assert_called_once()
                staged_index, = self.directory.glob(".index.faiss.*.tmp")
                staged_rag = self.build_rag.call_args.args[1]
                self.assertEqual(
                    set(self.directory.glob(".*.tmp")),
                    {staged_index, staged_rag},
                )
                self.assertEqual(faiss.read_index(str(staged_index)).ntotal, len(self.chunks))
                self.assertEqual(Path(source), staged_index)
                self.assertEqual(self.store.load_generation_metadata(staged_rag), {
                    "faiss_sha256": hashlib.sha256(staged_index.read_bytes()).hexdigest(),
                    "chunk_count": len(self.chunks),
                })
                self.assertEqual(self.fts.search_fts("newneedle", db_path=staged_rag), [0])
                self.assert_rag_contents(staged_rag, self.chunks)
            self.assertEqual(Path(source).parent, Path(destination).parent)
            replace(source, destination)
            self.assert_legacy_unchanged()

        with patch.object(self.builder.os, "replace", side_effect=publish) as replacements:
            self.builder.build_index()
        self.assertEqual(replacements.call_count, 2)
        self.assertEqual(
            [Path(call.args[1]) for call in replacements.call_args_list],
            [self.directory / "index.faiss", self.rag_path],
        )

        self.ingest.assert_called_once_with()
        self.chunk.assert_called_once_with(self.documents)
        self.embed.assert_called_once_with(self.chunks)
        self.assertIs(self.embed.call_args.args[0], self.chunks)
        self.build_rag.assert_called_once()
        self.assertIs(self.build_rag.call_args.args[0], self.chunks)
        published_sha = hashlib.sha256(
            (self.directory / "index.faiss").read_bytes()
        ).hexdigest()
        self.assertEqual(self.build_rag.call_args.kwargs, {"faiss_sha256": published_sha})
        self.assertEqual(self.store.load_generation_metadata(self.rag_path), {
            "faiss_sha256": published_sha,
            "chunk_count": len(self.chunks),
        })
        staged_rag = self.build_rag.call_args.args[1]
        self.assertEqual(staged_rag.parent, self.rag_path.parent)
        self.assertNotEqual(staged_rag, self.rag_path)

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
        self.assertEqual(
            {path.name for path in self.directory.iterdir()},
            {"index.faiss", "rag.db"},
        )

    def test_construction_completes_in_order_before_publication(self):
        events = []
        write_index = self.builder.faiss.write_index
        sha256_file = self.builder._sha256_file
        replace = self.builder.os.replace

        def write(index, path):
            write_index(index, path)
            events.append("write complete")

        def hash_staged(path):
            self.assertEqual(events, ["write complete"])
            digest = sha256_file(path)
            events.append("hash complete")
            return digest

        def build_rag(chunks, path, *, faiss_sha256):
            self.assertEqual(events, ["write complete", "hash complete"])
            self.store.build_rag_db(chunks, path, faiss_sha256=faiss_sha256)
            events.append("rag complete")

        def publish(source, destination):
            self.assertEqual(events[:3], ["write complete", "hash complete", "rag complete"])
            replace(source, destination)
            events.append("replace")

        self.build_rag.side_effect = build_rag
        with (
            patch.object(self.builder.faiss, "write_index", side_effect=write),
            patch.object(self.builder, "_sha256_file", side_effect=hash_staged),
            patch.object(self.builder.os, "replace", side_effect=publish),
        ):
            self.builder.build_index()
        self.assertEqual(events, [
            "write complete", "hash complete", "rag complete", "replace", "replace",
        ])

    def test_clean_first_build_never_generates_pickle(self):
        source = Path(self.builder.__spec__.origin).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                self.assertNotIn("pickle", [alias.name.split(".")[0] for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or "").split(".")[0], "pickle")
            elif isinstance(node, ast.Name):
                self.assertNotEqual(node.id, "pickle")

        replace = self.builder.os.replace

        def publish(source, destination):
            self.assertFalse(any(path.exists() for path in self.legacy_paths))
            self.assertEqual(list(self.directory.glob("*chunks.pkl*")), [])
            replace(source, destination)

        with (
            patch.object(self.builder, "_staged_artifact", wraps=self.builder._staged_artifact) as staging,
            patch.object(self.builder.os, "replace", side_effect=publish) as replacements,
            self.assertNoLogs(self.builder.logger, level="WARNING"),
        ):
            self.builder.build_index()
        self.assertEqual([call.args[0] for call in staging.call_args_list],
                         [self.directory / "index.faiss", self.rag_path])
        self.assertEqual(replacements.call_count, 2)
        self.assertEqual({path.name for path in self.directory.iterdir()},
                         {"index.faiss", "rag.db"})
        self.assertEqual(faiss.read_index(str(self.directory / "index.faiss")).ntotal,
                         len(self.chunks))
        self.assert_rag_contents(self.rag_path, self.chunks)

    def test_each_legacy_cleanup_failure_is_nonfatal_and_other_cleanup_continues(self):
        unlink = Path.unlink
        for failed_path in self.legacy_paths:
            with self.subTest(path=failed_path.name):
                self.seed_old_generation()
                self.build_rag.reset_mock()
                attempted = []

                def remove(path, *args, **kwargs):
                    if path.resolve() in self.legacy_paths:
                        attempted.append(path.resolve())
                        self.assertTrue(kwargs.get("missing_ok"))
                        self.assertEqual(replacements.call_count, 2)
                        self.assertEqual(
                            faiss.read_index(str(self.directory / "index.faiss")).ntotal,
                            len(self.chunks),
                        )
                        self.assert_rag_contents(self.rag_path, self.chunks)
                        if path.resolve() == failed_path:
                            raise PermissionError("injected legacy cleanup failure")
                    return unlink(path, *args, **kwargs)

                with (
                    patch.object(Path, "unlink", autospec=True, side_effect=remove),
                    patch.object(self.builder.os, "replace", wraps=self.builder.os.replace) as replacements,
                    self.assertLogs(self.builder.logger, level="WARNING") as logs,
                ):
                    self.builder.build_index()
                self.assertEqual(attempted, list(self.legacy_paths))
                self.build_rag.assert_called_once()
                self.assertEqual(replacements.call_count, 2)
                self.assertEqual(failed_path.read_bytes(), self.legacy_bytes)
                self.assertEqual({path.name for path in self.directory.iterdir()},
                                 {"index.faiss", "rag.db", failed_path.name})
                self.assertEqual(len(logs.records), 1)
                self.assertEqual(logs.records[0].levelno, self.builder.logging.WARNING)
                self.assertIn(failed_path.name, logs.records[0].getMessage())
                self.assertIn("injected legacy cleanup failure", logs.output[0])
                self.assertIn("Indexing complete", self.output.getvalue())

    def test_publication_failures_skip_legacy_cleanup_without_rollback(self):
        replace = self.builder.os.replace
        error = OSError("injected publication failure")
        for failed_operation in (1, 2):
            with self.subTest(operation=failed_operation):
                before = self.seed_old_generation()

                def publish(source, destination):
                    self.assert_legacy_unchanged()
                    if replacements.call_count == 1:
                        staged_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
                        self.assertEqual(self.build_rag.call_args.kwargs,
                                         {"faiss_sha256": staged_sha})
                        self.assertEqual(self.store.load_generation_metadata(
                            self.build_rag.call_args.args[1]
                        ), {"faiss_sha256": staged_sha, "chunk_count": len(self.chunks)})
                    if replacements.call_count == failed_operation:
                        raise error
                    replace(source, destination)

                with (
                    patch.object(self.builder.os, "replace", side_effect=publish) as replacements,
                    self.assertRaises(OSError) as raised,
                ):
                    self.builder.build_index()
                self.assertIs(raised.exception, error)
                self.assertEqual(replacements.call_count, failed_operation)
                self.assert_legacy_unchanged()
                self.assertEqual({path.name for path in self.directory.iterdir()},
                                 set(before) | {path.name for path in self.legacy_paths})
                self.assertEqual(self.rag_path.read_bytes(), before["rag.db"])
                self.assertEqual(self.store.load_generation_metadata(self.rag_path), self.old_metadata)
                if failed_operation == 1:
                    self.assert_old_generation_unchanged(before)
                else:
                    index = faiss.read_index(str(self.directory / "index.faiss"))
                    self.assertEqual(index.ntotal, len(self.chunks))
                    self.assertNotEqual((self.directory / "index.faiss").read_bytes(),
                                        before["index.faiss"])
                    published_sha = hashlib.sha256(
                        (self.directory / "index.faiss").read_bytes()
                    ).hexdigest()
                    self.assertEqual(published_sha, self.build_rag.call_args.kwargs["faiss_sha256"])
                    self.assertNotEqual(published_sha, self.old_metadata["faiss_sha256"])
                    self.assertEqual(index.ntotal, self.old_metadata["chunk_count"])
                    self.assert_rag_contents(self.rag_path, self.old_chunks)
                self.assertNotIn("Indexing complete", self.output.getvalue())

    def test_rag_build_failure_preserves_old_generation_and_cleans_staging(self):
        before = self.seed_old_generation()
        error = sqlite3.OperationalError("rag.db build failed")

        def fail_rag(chunks, db_path, *, faiss_sha256):
            self.assertIs(chunks, self.chunks)
            self.assertNotEqual(db_path, self.rag_path)
            self.assertEqual(db_path.parent, self.rag_path.parent)
            staged_index, = self.directory.glob(".index.faiss.*.tmp")
            self.assertEqual(faiss.read_index(str(staged_index)).ntotal, len(self.chunks))
            self.assertEqual(faiss_sha256, hashlib.sha256(staged_index.read_bytes()).hexdigest())
            self.assertEqual(self.artifact_bytes(), before)
            self.assert_legacy_unchanged()
            # Fail inside the real SQLite transaction, after inserting chunks.
            with patch.object(self.store, "_rebuild_fts", side_effect=error):
                try:
                    self.store.build_rag_db(chunks, db_path, faiss_sha256=faiss_sha256)
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

    def test_hash_failure_preserves_old_generation_and_cleans_staging(self):
        before = self.seed_old_generation()
        error = OSError("staged FAISS hash failed")

        def fail_hash(path):
            self.assertNotEqual(path, self.directory / "index.faiss")
            self.assertEqual(faiss.read_index(str(path)).ntotal, len(self.chunks))
            self.assertEqual(self.artifact_bytes(), before)
            staged_rag, = self.directory.glob(".rag.db.*.tmp")
            for suffix in ("-journal", "-wal", "-shm"):
                Path(str(staged_rag) + suffix).write_bytes(b"staging cleanup sentinel")
            raise error

        with (
            patch.object(self.builder, "_sha256_file", side_effect=fail_hash) as hashing,
            patch.object(self.builder.os, "replace") as replacements,
            patch.object(Path, "unlink", autospec=True, side_effect=Path.unlink) as unlink,
            self.assertRaises(OSError) as raised,
        ):
            self.builder.build_index()
        self.assertIs(raised.exception, error)
        hashing.assert_called_once()
        self.build_rag.assert_not_called()
        replacements.assert_not_called()
        self.assertTrue(all(call.args[0].resolve() not in self.legacy_paths
                            for call in unlink.call_args_list))
        self.assert_old_generation_unchanged(before)
        self.assertNotIn("Indexing complete", self.output.getvalue())

    def test_metadata_failure_preserves_old_generation_and_cleans_staging(self):
        before = self.seed_old_generation()
        error = sqlite3.OperationalError("generation metadata insert failed")
        with (
            patch.object(self.store, "_insert_generation_metadata", side_effect=error),
            patch.object(self.builder.os, "replace") as replacements,
            self.assertRaises(sqlite3.OperationalError) as raised,
        ):
            self.builder.build_index()
        self.assertIs(raised.exception, error)
        self.build_rag.assert_called_once()
        replacements.assert_not_called()
        self.assert_old_generation_unchanged(before)
        self.assertNotIn("Indexing complete", self.output.getvalue())

    def test_earlier_construction_failures_preserve_old_generation(self):
        before = self.seed_old_generation()
        error = OSError("injected construction failure")

        def fail_faiss(index, path):
            Path(path).write_bytes(b"partial FAISS")
            raise error

        failures = [
            (self.builder, "embed_chunks", error),
            (self.builder.faiss, "IndexFlatIP", error),
            (self.builder.faiss, "normalize_L2", error),
            (self.builder.faiss, "write_index", fail_faiss),
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

    def test_hash_failure_on_first_build_publishes_nothing(self):
        self.seed_legacy_files()
        error = OSError("first staged FAISS hash failed")
        with (
            patch.object(self.builder, "_sha256_file", side_effect=error),
            patch.object(self.builder.os, "replace") as replacements,
            self.assertRaises(OSError) as raised,
        ):
            self.builder.build_index()
        self.assertIs(raised.exception, error)
        self.build_rag.assert_not_called()
        replacements.assert_not_called()
        self.assert_legacy_unchanged()
        self.assertEqual(set(self.directory.iterdir()), set(self.legacy_paths))

    def test_rag_failure_on_first_build_publishes_nothing(self):
        self.seed_legacy_files()
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
        self.assert_legacy_unchanged()
        self.assertEqual(set(self.directory.iterdir()), set(self.legacy_paths))

    def test_relative_paths_keep_existing_base_directories(self):
        src_dir = self.directory / "src"
        src_dir.mkdir()
        cwd = self.directory / "working"
        cwd.mkdir()
        legacy_chunks = src_dir / "chunks.pkl"
        legacy_fts = cwd / "fts_index.db"
        wrong_chunks = cwd / "chunks.pkl"
        wrong_fts = src_dir / "fts_index.db"
        for path in (legacy_chunks, legacy_fts, wrong_chunks, wrong_fts):
            path.write_bytes(self.legacy_bytes)
        with (
            chdir(cwd),
            patch.object(self.builder, "__file__", str(src_dir / "rag/build_index.py")),
            patch.object(self.builder, "FAISS_INDEX_PATH", "index.faiss"),
            patch.object(self.builder, "RAG_DB_PATH", "storage/rag.db"),
        ):
            self.builder.build_index()
        self.assertEqual(
            {str(path.relative_to(self.directory)) for path in self.directory.rglob("*") if path.is_file()},
            {"src/index.faiss", "working/storage/rag.db",
             "working/chunks.pkl", "src/fts_index.db"},
        )
        for path in (wrong_chunks, wrong_fts):
            self.assertEqual(path.read_bytes(), self.legacy_bytes)
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
