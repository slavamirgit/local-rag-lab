from fastmcp import FastMCP
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DOCUMENTS_DIR

mcp = FastMCP("doc-tools", version="1.0.0")


@mcp.tool
def read_document(file_path: str) -> str:
    """Reads a document from the knowledge base."""
    try:
        root = Path(DOCUMENTS_DIR).resolve()
        path = Path(file_path).resolve()
        # Compare path components after resolving traversal and symlinks.
        if not path.is_relative_to(root):
            return f"Error: Access denied. File must be in {DOCUMENTS_DIR}"
        
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@mcp.tool
def list_documents() -> str:
    """Lists all available documents in the knowledge base."""
    try:
        base_dir = Path(DOCUMENTS_DIR)
        if not base_dir.exists():
            return f"Error: Documents directory {DOCUMENTS_DIR} does not exist"
        
        documents = []
        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                documents.append(str(path.relative_to(base_dir)))
        
        if not documents:
            return "No documents found in the knowledge base."
        
        return "\n".join(f"- {doc}" for doc in sorted(documents))
    except Exception as e:
        return f"Error listing documents: {str(e)}"


@mcp.tool
def search_documents(query: str) -> str:
    """Searches for documents by name (case-insensitive)."""
    try:
        base_dir = Path(DOCUMENTS_DIR)
        if not base_dir.exists():
            return f"Error: Documents directory {DOCUMENTS_DIR} does not exist"
        
        query_lower = query.lower()
        matches = []
        
        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                if query_lower in path.name.lower():
                    matches.append(str(path.relative_to(base_dir)))
        
        if not matches:
            return f"No documents found matching '{query}'"
        
        return "\n".join(f"- {doc}" for doc in sorted(matches))
    except Exception as e:
        return f"Error searching documents: {str(e)}"


if __name__ == "__main__":
    # Run MCP server (stdio)
    mcp.run()
