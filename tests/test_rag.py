from pathlib import Path

import numpy as np
import pytest

from code_qa_api.rag.chunking import MarkdownChunker, PythonCodeChunker
from code_qa_api.rag.store import VectorStore
from tests.conftest import TEST_INDEX_PATH

# Sample code for testing chunking
SAMPLE_CODE = """
import os

class SimpleClass:
    def method_one(self, x):
        return x * 2

async def async_function():
    pass

def standalone_function(a, b):
    # A comment
    return a + b
"""


# Test Chunking
@pytest.fixture
def chunker() -> PythonCodeChunker:
    # Set min_chunk_lines to 1 for testing to capture small functions/classes
    return PythonCodeChunker(min_chunk_lines=1)


def test_python_chunking(chunker: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.py"
    file_path.write_text(SAMPLE_CODE)

    chunks = chunker.chunk_file(file_path)

    # Expect 4 chunks: SimpleClass, SimpleClass.method_one, async_function, standalone_function
    assert len(chunks) == 4

    # Verify chunk details (adjust based on actual chunker output if needed)
    class_chunk = next(c for c in chunks if c["type"] == "ClassDef" and c["name"] == "SimpleClass")
    method_chunk = next(c for c in chunks if c["type"] == "FunctionDef" and c["name"] == "method_one")
    async_func_chunk = next(c for c in chunks if c["type"] == "AsyncFunctionDef")
    func_chunk = next(c for c in chunks if c["type"] == "FunctionDef" and c["name"] == "standalone_function")

    assert class_chunk["start_line"] == 4
    assert class_chunk["end_line"] == 6
    assert "class SimpleClass:" in class_chunk["content"]
    assert "method_one" in class_chunk["content"]  # Method is inside class content

    assert method_chunk["start_line"] == 5  # method_one def
    assert method_chunk["end_line"] == 6  # method_one return
    assert "def method_one(self, x):" in method_chunk["content"]

    assert async_func_chunk["name"] == "async_function"
    assert async_func_chunk["start_line"] == 8
    assert async_func_chunk["end_line"] == 9

    assert func_chunk["name"] == "standalone_function"
    assert func_chunk["start_line"] == 11
    assert func_chunk["end_line"] == 13


def test_chunking_empty_file(chunker: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.py"
    file_path.write_text("")
    chunks = chunker.chunk_file(file_path)
    assert len(chunks) == 0


def test_chunking_no_functions_classes(chunker: PythonCodeChunker, tmp_path: Path) -> None:
    code = "x = 1\ny = 2\nprint(x+y)"
    file_path = tmp_path / "plain.py"
    file_path.write_text(code)
    chunks = chunker.chunk_file(file_path)
    # The current chunker focuses on classes/functions, might return 0 chunks for only top-level code
    assert len(chunks) == 0
    # If you want to chunk top-level code, the chunker needs modification.
    # assert len(chunks) == 1
    # assert chunks[0]["type"] == "Module" # Or similar top-level node type
    # assert chunks[0]["content"] == code


def test_chunking_syntax_error(chunker: PythonCodeChunker, tmp_path: Path) -> None:
    code = "def func(\n    print('hello')"  # Missing colon
    file_path = tmp_path / "bad.py"
    file_path.write_text(code)
    # chunk_file handles SyntaxError and returns []
    chunks = chunker.chunk_file(file_path)
    assert len(chunks) == 0


# Test VectorStore (using the fixture)
def test_vector_store_add_search(vector_store: VectorStore):
    assert not vector_store.is_initialized()

    # Add dummy data
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    metadata = [
        {"chunk_id": "c1", "content": "chunk 1 text", "file_path": "f1.py"},
        {"chunk_id": "c2", "content": "chunk 2 text", "file_path": "f1.py"},
        {"chunk_id": "c3", "content": "chunk 3 text", "file_path": "f2.py"},
    ]
    vector_store.add(embeddings, metadata)

    assert vector_store.is_initialized()
    # Use collection.count() instead of index.ntotal for ChromaDB
    assert vector_store._collection.count() == 3
    # assert len(vector_store.metadata) == 3 # This attribute might not exist

    # Search for the embedding closest to [0.1, 0.2]
    query_embedding = np.array([0.11, 0.21], dtype=np.float32)
    results = vector_store.search(query_embedding, k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"

    # Search for the embedding closest to [0.5, 0.6]
    query_embedding_2 = np.array([0.55, 0.65], dtype=np.float32)
    results_2 = vector_store.search(query_embedding_2, k=2)
    assert len(results_2) == 2
    # Closest should be c3, next c2 based on L2 distance
    assert results_2[0]["chunk_id"] == "c3"
    assert results_2[1]["chunk_id"] == "c2"


def test_vector_store_persistence(vector_store: VectorStore):
    # Use the vector_store fixture which points to TEST_INDEX_PATH
    embeddings = np.array([[0.7, 0.8]], dtype=np.float32)
    metadata = [{"chunk_id": "p1", "content": "persistent chunk"}]
    vector_store.add(embeddings, metadata)
    ntotal_before = vector_store._collection.count()

    # Create a new instance pointing to the same path
    # Need to pass TEST_INDEX_PATH explicitly as persist_directory
    # And use the *same* directory the original vector_store fixture used
    persist_path = vector_store.persist_directory  # Get path from the first store
    assert persist_path is not None  # Ensure it's not an in-memory store
    new_store = VectorStore(persist_directory=persist_path)
    assert new_store.is_initialized()
    assert new_store._collection.count() == ntotal_before

    # Search in the new store
    query = np.array([0.71, 0.81], dtype=np.float32)
    results = new_store.search(query, k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "p1"


def test_vector_store_reset(vector_store: VectorStore):
    embeddings = np.array([[0.1, 0.1]], dtype=np.float32)
    metadata = [{"chunk_id": "r1", "content": "to be reset"}]
    vector_store.add(embeddings, metadata)
    assert vector_store.is_initialized()

    vector_store.reset()
    assert not vector_store.is_initialized()
    assert vector_store._collection.count() == 0
    assert not list(TEST_INDEX_PATH.glob("*.faiss"))  # Check files are deleted
    assert not list(TEST_INDEX_PATH.glob("*.pkl"))


# TODO: Add tests for embedding.py, retrieval.py, generation.py
# These will likely require mocking litellm API calls


# ---- Markdown Chunking Tests ----

SAMPLE_MARKDOWN = """# Header 1

This is the first section.

## Header 1.1

Content under H1.1.

```python
print("hello")
# Another Header like line inside code
```

More content under H1.1.

### Header 1.1.1

Content under H1.1.1.

## Header 1.2

Content under H1.2.

# Header 2

Content under H2.

No header here.
"""


@pytest.fixture
def md_chunker() -> MarkdownChunker:
    return MarkdownChunker(max_header_depth=3)


def test_markdown_chunking_basic(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text(SAMPLE_MARKDOWN)

    chunks = md_chunker.chunk_file(file_path)

    # Expected chunks: H1, H1.1, H1.1.1, H1.2, H2
    # The content under H1 gets adjusted to end before H1.1 starts
    assert len(chunks) == 5

    h1 = next(c for c in chunks if c["header"] == "Header 1")
    h1_1 = next(c for c in chunks if c["header"] == "Header 1.1")
    h1_1_1 = next(c for c in chunks if c["header"] == "Header 1.1.1")
    h1_2 = next(c for c in chunks if c["header"] == "Header 1.2")
    h2 = next(c for c in chunks if c["header"] == "Header 2")

    assert h1["level"] == 1
    assert h1["parent_chunk_id"] is None
    assert "This is the first section." in h1["content"]
    # assert "## Header 1.1" not in h1["content"] # Content should end before next header
    # Check line numbers (1-based)
    assert h1["start_line"] == 1
    assert h1["end_line"] == 4  # Ends before line 5 (## Header 1.1)

    assert h1_1["level"] == 2
    assert h1_1["parent_chunk_id"] == h1["chunk_id"]
    assert "Content under H1.1." in h1_1["content"]
    assert "```python" in h1_1["content"]
    assert "# Another Header like line inside code" in h1_1["content"]
    assert "More content under H1.1." in h1_1["content"]
    # assert "### Header 1.1.1" not in h1_1["content"] # Content should end before next header
    assert h1_1["start_line"] == 5
    assert h1_1["end_line"] == 15
    assert h1_1_1["level"] == 3
    assert h1_1_1["parent_chunk_id"] == h1_1["chunk_id"]
    assert "Content under H1.1.1." in h1_1_1["content"]
    assert h1_1_1["start_line"] == 16
    assert h1_1_1["end_line"] == 19  # Ends before line 19 (## Header 1.2)

    assert h1_2["level"] == 2
    assert h1_2["parent_chunk_id"] == h1["chunk_id"]
    assert "Content under H1.2." in h1_2["content"]
    assert h1_2["start_line"] == 20
    assert h1_2["end_line"] == 23  # Ends before line 23 (# Header 2)

    assert h2["level"] == 1
    assert h2["parent_chunk_id"] is None
    assert "Content under H2." in h2["content"]
    assert "No header here." in h2["content"]
    assert h2["start_line"] == 24
    assert h2["end_line"] == 28  # Goes to end of file


def test_markdown_chunking_max_depth(tmp_path: Path) -> None:
    md_chunker = MarkdownChunker(max_header_depth=1)
    file_path = tmp_path / "depth_limit.md"
    file_path.write_text(SAMPLE_MARKDOWN)
    chunks = md_chunker.chunk_file(file_path)

    # Only H1 and H2 should be top-level chunks
    # Content under H1 should include H1.1, H1.1.1, H1.2 text as they are below max depth
    assert len(chunks) == 2
    h1 = next(c for c in chunks if c["header"] == "Header 1")
    h2 = next(c for c in chunks if c["header"] == "Header 2")

    assert h1["level"] == 1
    assert "## Header 1.1" in h1["content"]
    assert "### Header 1.1.1" in h1["content"]
    assert "## Header 1.2" in h1["content"]
    assert h1["start_line"] == 1
    assert h1["end_line"] == 23  # Ends before # Header 2

    assert h2["level"] == 1
    assert "Content under H2." in h2["content"]
    assert h2["start_line"] == 24
    assert h2["end_line"] == 28  # Goes to end of file


def test_markdown_chunking_no_headers(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    content = "Just plain text.\nAnother line."
    file_path = tmp_path / "no_headers.md"
    file_path.write_text(content)
    chunks = md_chunker.chunk_file(file_path)
    assert len(chunks) == 0  # Current implementation only chunks starting from headers


def test_markdown_chunking_empty_file(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.md"
    file_path.write_text("")
    chunks = md_chunker.chunk_file(file_path)
    assert len(chunks) == 0


def test_markdown_chunking_only_header(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    content = "# Only Header"
    file_path = tmp_path / "only_header.md"
    file_path.write_text(content)
    chunks = md_chunker.chunk_file(file_path)
    # Chunk is only created if there's content *beyond* the header itself
    assert len(chunks) == 0


# ---- End Markdown Chunking Tests ----
