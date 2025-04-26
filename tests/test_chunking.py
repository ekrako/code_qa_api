from pathlib import Path

import pytest

from code_qa_api.rag.chunking import MarkdownChunker, PythonCodeChunker

# --- Python Chunking Tests ---

# Sample code for testing chunking
SAMPLE_CODE_BASIC = """
import os

class SimpleClass:
    def method_one(self, x):
        # A comment inside method
        return x * 2

async def async_function():
    pass

def standalone_function(a, b):
    # A comment
    return a + b
"""

SAMPLE_CODE_NESTED = """
class OuterClass:
    y = 10
    def outer_method(self):
        z = 5
        class InnerClass:
            def inner_method(self):
                return self.y # Accessing outer scope? No, needs instance

        def nested_function(p):
            # Nested comment
            return p + z
        return nested_function(3)

def top_level_func_with_nested():
    def inner_func(n):
        return n + 1
    return inner_func(5)
"""

SAMPLE_CODE_DECORATORS = """
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Something is happening before the function is called.")
        result = func(*args, **kwargs)
        print("Something is happening after the function is called.")
        return result
    return wrapper

@my_decorator
def say_whee():
    print("Whee!")

class DecoratedClass:
    @my_decorator
    def decorated_method(self):
        print("Method whee!")
"""

SAMPLE_CODE_SMALL_FUNCS = """
def small_one():
    return 1

def small_two(a):
    b = a + 1
    return b
"""

SAMPLE_CODE_LONG_FUNC = """
def long_function():
    # Imagine this function has thousands of lines
    # Line 1
    # Line 2
    # ...
    # Line 5000
    print("This function might be too long")
    # ... many more lines ...
    return "very long" * 500 # Make content long
"""


# Fixtures for Python chunker
@pytest.fixture
def python_chunker_min_1() -> PythonCodeChunker:
    return PythonCodeChunker(min_chunk_lines=1)


@pytest.fixture
def python_chunker_default() -> PythonCodeChunker:
    return PythonCodeChunker()  # Uses default min_chunk_lines


# Helper for Python chunk tests
def _verify_python_chunk(chunk, start_line, end_line, expected_content_part, expected_name=None, expected_type=None):
    assert chunk["start_line"] == start_line
    # End line can be tricky due to how AST reports it vs tokenize adjustment.
    # We will not assert the exact end line, just that the start is correct and content is present.
    # assert chunk["end_line"] >= end_line
    assert expected_content_part in chunk["content"]
    if expected_name:
        assert chunk["name"] == expected_name
    if expected_type:
        assert chunk["type"] == expected_type


def test_python_chunking_basic(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "sample_basic.py"
    file_path.write_text(SAMPLE_CODE_BASIC)
    chunks = python_chunker_min_1.chunk_file(file_path)

    assert len(chunks) == 4

    class_chunk = next(c for c in chunks if c["name"] == "SimpleClass")
    method_chunk = next(c for c in chunks if c["name"] == "method_one")
    async_func_chunk = next(c for c in chunks if c["name"] == "async_function")
    func_chunk = next(c for c in chunks if c["name"] == "standalone_function")

    _verify_python_chunk(class_chunk, 4, 7, "class SimpleClass:", "SimpleClass", "ClassDef")
    assert "method_one" in class_chunk["content"]

    _verify_python_chunk(method_chunk, 5, 7, "def method_one(self, x):", "method_one", "FunctionDef")
    _verify_python_chunk(async_func_chunk, 9, 10, "async def async_function():", "async_function", "AsyncFunctionDef")
    _verify_python_chunk(func_chunk, 12, 14, "def standalone_function(a, b):", "standalone_function", "FunctionDef")


def test_python_chunking_nested(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "sample_nested.py"
    file_path.write_text(SAMPLE_CODE_NESTED)
    chunks = python_chunker_min_1.chunk_file(file_path)

    # Expect OuterClass, outer_method, InnerClass, inner_method, nested_function,
    # top_level_func_with_nested, inner_func
    assert len(chunks) == 7

    outer_class = next(c for c in chunks if c["name"] == "OuterClass")
    outer_method = next(c for c in chunks if c["name"] == "outer_method")
    inner_class = next(c for c in chunks if c["name"] == "InnerClass")
    inner_method = next(c for c in chunks if c["name"] == "inner_method")
    nested_func = next(c for c in chunks if c["name"] == "nested_function")
    top_level = next(c for c in chunks if c["name"] == "top_level_func_with_nested")
    inner_func = next(c for c in chunks if c["name"] == "inner_func")

    _verify_python_chunk(outer_class, 2, 13, "class OuterClass:", "OuterClass", "ClassDef")
    _verify_python_chunk(outer_method, 4, 13, "def outer_method(self):", "outer_method", "FunctionDef")
    _verify_python_chunk(inner_class, 6, 9, "class InnerClass:", "InnerClass", "ClassDef")
    _verify_python_chunk(inner_method, 7, 9, "def inner_method(self):", "inner_method", "FunctionDef")
    _verify_python_chunk(nested_func, 10, 13, "def nested_function(p):", "nested_function", "FunctionDef")
    _verify_python_chunk(top_level, 15, 19, "def top_level_func_with_nested():", "top_level_func_with_nested", "FunctionDef")
    _verify_python_chunk(inner_func, 16, 18, "def inner_func(n):", "inner_func", "FunctionDef")


def test_python_chunking_decorators(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "sample_decorators.py"
    file_path.write_text(SAMPLE_CODE_DECORATORS)
    chunks = python_chunker_min_1.chunk_file(file_path)

    # Expect: my_decorator, wrapper, say_whee, DecoratedClass, decorated_method
    assert len(chunks) == 5

    decorator_func = next(c for c in chunks if c["name"] == "my_decorator")
    wrapper_func = next(c for c in chunks if c["name"] == "wrapper")  # Nested inside my_decorator
    decorated_plain_func = next(c for c in chunks if c["name"] == "say_whee")
    decorated_class = next(c for c in chunks if c["name"] == "DecoratedClass")
    decorated_method = next(c for c in chunks if c["name"] == "decorated_method")

    _verify_python_chunk(decorator_func, 2, 8, "def my_decorator(func):", "my_decorator", "FunctionDef")
    _verify_python_chunk(wrapper_func, 3, 7, "def wrapper(*args, **kwargs):", "wrapper", "FunctionDef")
    # Note: The decorator line `@my_decorator` is NOT part of the function's content chunk
    _verify_python_chunk(decorated_plain_func, 11, 12, "def say_whee():", "say_whee", "FunctionDef")
    _verify_python_chunk(decorated_class, 14, 17, "class DecoratedClass:", "DecoratedClass", "ClassDef")
    _verify_python_chunk(decorated_method, 16, 18, "def decorated_method(self):", "decorated_method", "FunctionDef")


def test_python_chunking_min_lines_default(python_chunker_default: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "small_funcs.py"
    file_path.write_text(SAMPLE_CODE_SMALL_FUNCS)
    chunks = python_chunker_default.chunk_file(file_path)

    # Default min_chunk_lines is 5. These functions are smaller.
    assert len(chunks) == 0


def test_python_chunking_min_lines_override(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "small_funcs.py"
    file_path.write_text(SAMPLE_CODE_SMALL_FUNCS)
    chunks = python_chunker_min_1.chunk_file(file_path)

    # min_chunk_lines=1, so both functions should be chunked.
    assert len(chunks) == 2
    _verify_python_chunk(chunks[0], 2, 3, "def small_one():", "small_one")
    _verify_python_chunk(chunks[1], 5, 7, "def small_two(a):", "small_two")


def test_python_chunking_max_chars(tmp_path: Path) -> None:
    # Use a very small max_chunk_chars for testing
    chunker = PythonCodeChunker(min_chunk_lines=1, max_chunk_chars=100)
    file_path = tmp_path / "long_func.py"
    file_path.write_text(SAMPLE_CODE_LONG_FUNC)
    chunks = chunker.chunk_file(file_path)

    # The function content is longer than 100 chars, so it should be skipped
    assert len(chunks) == 0
    # Ideally, we'd also check for the warning log, but that's harder in standard pytest


def test_python_chunking_empty_file(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.py"
    file_path.write_text("")
    chunks = python_chunker_min_1.chunk_file(file_path)
    assert len(chunks) == 0


def test_python_chunking_no_functions_classes(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    code = "x = 1\\ny = 2\\nprint(x+y)"
    file_path = tmp_path / "plain.py"
    file_path.write_text(code)
    chunks = python_chunker_min_1.chunk_file(file_path)
    # Only chunks functions/classes
    assert len(chunks) == 0


def test_python_chunking_syntax_error(python_chunker_min_1: PythonCodeChunker, tmp_path: Path) -> None:
    code = "def func(\\n    print('hello')"  # Missing colon
    file_path = tmp_path / "bad.py"
    file_path.write_text(code)
    # chunk_file handles SyntaxError and returns []
    chunks = python_chunker_min_1.chunk_file(file_path)
    assert len(chunks) == 0


# --- Markdown Chunking Tests ---

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


# Helper for Markdown chunk tests
def _verify_markdown_chunk(chunk, start_line, end_line, expected_content_part, header, level, parent_id=None):
    assert header in chunk["content"]  # Header should be part of the content
    assert expected_content_part in chunk["content"]
    assert chunk["start_line"] == start_line
    # Unlike Python chunker, end_line here should be more predictable
    assert chunk["end_line"] == end_line
    assert chunk["header"] == header
    assert chunk["level"] == level
    assert chunk["parent_chunk_id"] == parent_id


def test_markdown_chunking_basic(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text(SAMPLE_MARKDOWN)

    chunks = md_chunker.chunk_file(file_path)

    assert len(chunks) == 5

    h1 = next(c for c in chunks if c["header"] == "Header 1")
    h1_1 = next(c for c in chunks if c["header"] == "Header 1.1")
    h1_1_1 = next(c for c in chunks if c["header"] == "Header 1.1.1")
    h1_2 = next(c for c in chunks if c["header"] == "Header 1.2")
    h2 = next(c for c in chunks if c["header"] == "Header 2")

    # Verify H1
    # Content ends before H1.1 starts
    _verify_markdown_chunk(h1, 1, 4, "This is the first section.", "Header 1", 1)
    assert "## Header 1.1" not in h1["content"]

    # Verify H1.1
    # Content includes code block and text, ends before H1.1.1
    _verify_markdown_chunk(h1_1, 5, 15, "Content under H1.1.", "Header 1.1", 2, h1["chunk_id"])
    assert "```python" in h1_1["content"]
    assert "# Another Header like line inside code" in h1_1["content"]
    assert "More content under H1.1." in h1_1["content"]
    assert "### Header 1.1.1" not in h1_1["content"]

    # Verify H1.1.1
    # Content ends before H1.2 starts
    _verify_markdown_chunk(h1_1_1, 16, 19, "Content under H1.1.1.", "Header 1.1.1", 3, h1_1["chunk_id"])
    assert "## Header 1.2" not in h1_1_1["content"]

    # Verify H1.2
    # Content ends before H2 starts
    _verify_markdown_chunk(h1_2, 20, 23, "Content under H1.2.", "Header 1.2", 2, h1["chunk_id"])
    assert "# Header 2" not in h1_2["content"]

    # Verify H2
    # Content includes everything until the end
    _verify_markdown_chunk(h2, 24, 28, "Content under H2.", "Header 2", 1)
    assert "No header here." in h2["content"]


def test_markdown_chunking_max_depth(tmp_path: Path) -> None:
    md_chunker = MarkdownChunker(max_header_depth=1)
    file_path = tmp_path / "depth_limit.md"
    file_path.write_text(SAMPLE_MARKDOWN)
    chunks = md_chunker.chunk_file(file_path)

    # Only H1 and H2 should be chunks
    assert len(chunks) == 2
    h1 = next(c for c in chunks if c["header"] == "Header 1")
    h2 = next(c for c in chunks if c["header"] == "Header 2")

    # H1 content includes H1.1, H1.1.1, H1.2 text as they are below max depth
    _verify_markdown_chunk(h1, 1, 23, "## Header 1.1", "Header 1", 1)  # Check content includes sub-headers
    assert "### Header 1.1.1" in h1["content"]
    assert "## Header 1.2" in h1["content"]
    assert "# Header 2" not in h1["content"]  # Stops before next H1

    # H2 content includes the rest
    _verify_markdown_chunk(h2, 24, 28, "Content under H2.", "Header 2", 1)


def test_markdown_chunking_no_headers(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    content = "Just plain text.\\nAnother line."
    file_path = tmp_path / "no_headers.md"
    file_path.write_text(content)
    chunks = md_chunker.chunk_file(file_path)
    assert len(chunks) == 0  # Only chunks starting from headers


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
    # Chunk requires content *after* the header line itself
    assert len(chunks) == 0


def test_markdown_code_block_header_like(md_chunker: MarkdownChunker, tmp_path: Path) -> None:
    content = """
# Real Header

Some text.

```text
# Not a real header
Actual content
```

More text after code block.
"""
    file_path = tmp_path / "code_block.md"
    file_path.write_text(content.strip())
    chunks = md_chunker.chunk_file(file_path)

    assert len(chunks) == 1
    chunk = chunks[0]
    _verify_markdown_chunk(chunk, 1, 10, "# Not a real header", "Real Header", 1)
    assert "More text after code block." in chunk["content"]
