from contextlib import chdir
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

# Other application imports add src to sys.path. Keep its local mcp package
# from shadowing FastMCP's mcp dependency while importing the real server.
src_dir = Path(__file__).resolve().parents[1] / "src"
with patch.object(sys, "path", [p for p in sys.path if Path(p).resolve() != src_dir]):
    from src.mcp import server


class ReadDocumentTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.base = Path(self.directory)
        self.root = self.base / "docs"
        self.root.mkdir()
        self.nested = self.root / "nested"
        self.nested.mkdir()
        self.allowed = self.root / "allowed.txt"
        self.allowed.write_text("Root document: café", encoding="utf-8")
        (self.nested / "allowed.txt").write_text("Nested document", encoding="utf-8")
        self.outside = self.base / "outside.txt"
        self.outside.write_text("Outside document", encoding="utf-8")
        self.sibling = self.base / "docs-other"
        self.sibling.mkdir()
        (self.sibling / "probe.txt").write_text("Sibling document", encoding="utf-8")
        self.enterContext(patch.object(server, "DOCUMENTS_DIR", str(self.root)))

    def assert_denied(self, path):
        self.assertEqual(
            server.read_document(str(path)),
            f"Error: Access denied. File must be in {server.DOCUMENTS_DIR}",
        )

    def symlink(self, link, target):
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Symlinks unavailable: {exc}")

    def test_file_in_root_is_readable(self):
        self.assertEqual(server.read_document(str(self.allowed)), "Root document: café")

    def test_nested_file_is_readable(self):
        self.assertEqual(
            server.read_document(str(self.nested / "allowed.txt")), "Nested document"
        )

    def test_relative_root_and_candidate_use_working_directory(self):
        with chdir(self.base), patch.object(server, "DOCUMENTS_DIR", "./docs"):
            self.assertEqual(server.read_document("docs/allowed.txt"), "Root document: café")
            self.assert_denied("docs-other/probe.txt")

    def test_parent_traversal_outside_is_rejected(self):
        with chdir(self.root):
            self.assert_denied("../outside.txt")
            self.assert_denied("../docs-other/probe.txt")

    def test_absolute_outside_path_is_rejected(self):
        self.assert_denied(self.outside)

    def test_sibling_prefix_path_is_rejected(self):
        self.assert_denied(self.sibling / "probe.txt")

    def test_symlink_to_outside_file_is_rejected(self):
        link = self.root / "escape.txt"
        self.symlink(link, self.outside)
        self.assert_denied(link)

    def test_symlink_to_outside_directory_is_rejected(self):
        link = self.root / "escape"
        self.symlink(link, self.sibling)
        self.assert_denied(link / "probe.txt")

    def test_symlink_within_root_is_readable(self):
        link = self.root / "alias.txt"
        self.symlink(link, self.allowed)
        self.assertEqual(server.read_document(str(link)), "Root document: café")

    def test_configured_root_symlink_is_resolved(self):
        link = self.base / "root-alias"
        self.symlink(link, self.root)
        with patch.object(server, "DOCUMENTS_DIR", str(link)):
            self.assertEqual(server.read_document(str(self.allowed)), "Root document: café")
            self.assertEqual(
                server.read_document(str(link / "allowed.txt")), "Root document: café"
            )
            self.assert_denied(self.sibling / "probe.txt")

    def test_legitimate_punctuation_and_normalized_paths_are_allowed(self):
        for name in ("release..notes.txt", "policy v1.2 - final.txt", ".hidden-file.txt"):
            with self.subTest(name=name):
                path = self.root / name
                path.write_text(name, encoding="utf-8")
                self.assertEqual(server.read_document(str(path)), name)
        self.assertEqual(
            server.read_document(str(self.nested / ".." / "allowed.txt")),
            "Root document: café",
        )

    def test_nonexistent_file_keeps_file_not_found_response(self):
        path = self.root / "missing.txt"
        self.assertEqual(
            server.read_document(str(path)), f"Error: File not found: {path}"
        )

    def test_directory_keeps_read_error_response(self):
        self.assertTrue(server.read_document(str(self.nested)).startswith("Error reading file: "))

    def test_invalid_utf8_keeps_read_error_response(self):
        path = self.root / "invalid.txt"
        path.write_bytes(b"\xff")
        result = server.read_document(str(path))
        self.assertTrue(result.startswith("Error reading file: "))
        self.assertIn("utf-8", result)

    def test_list_and_search_keep_existing_results(self):
        self.assertEqual(server.list_documents(), "- allowed.txt\n- nested/allowed.txt")
        self.assertEqual(server.search_documents("ALLOW"), "- allowed.txt\n- nested/allowed.txt")
        self.assertEqual(server.search_documents("probe"), "No documents found matching 'probe'")


if __name__ == "__main__":
    unittest.main()
