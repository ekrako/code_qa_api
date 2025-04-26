import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Any

from code_qa_api.utils.file_handler import read_file_content


class PythonCodeChunker:
    def __init__(self, min_chunk_lines: int = 5, max_chunk_chars: int = 50000):
        self.min_chunk_lines = min_chunk_lines
        self.max_chunk_chars = max_chunk_chars

    def _get_node_content(self, node: ast.AST, lines: list[str]) -> str:
        start_line = node.lineno - 1  # type: ignore[attr-defined]
        end_line = getattr(node, "end_lineno", start_line + 1)  # type: ignore[attr-defined]

        # Use tokenize to get more accurate end line for multi-line statements
        # ast node end_lineno might not be precise enough for complex statements
        try:
            if tokens := list(tokenize.generate_tokens(io.StringIO("\n".join(lines[start_line:end_line])).readline)):
                # Sometimes end_lineno points *after* the last line of the node
                # Find the actual last token's end line
                actual_end_line = max(t.end[0] for t in tokens)
                end_line = start_line + actual_end_line
        except (tokenize.TokenError, IndentationError):
            # Fallback if tokenization fails (e.g., syntax errors in slice)
            pass
        except IndexError:
            # Handle case where start_line might be out of bounds temporarily
            pass

        return "\n".join(lines[start_line:end_line])

    def chunk_file(self, file_path: Path) -> list[dict[str, Any]]:
        content = read_file_content(file_path)
        if not content:
            return []

        lines = content.splitlines()
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            print(f"Warning: Skipping file {file_path} due to SyntaxError: {e}")
            return []

        chunks = []
        chunk_id_counter = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", start_line)
                node_content = self._get_node_content(node, lines)

                if not node_content.strip():  # Skip empty nodes
                    continue

                num_lines = end_line - start_line + 1
                num_chars = len(node_content)

                if num_lines >= self.min_chunk_lines and num_chars <= self.max_chunk_chars:
                    chunk_id_counter += 1
                    chunk = {
                        "chunk_id": chunk_id_counter,
                        "file_path": str(file_path),
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": node_content,
                        "type": type(node).__name__,
                        "name": getattr(node, "name", "N/A"),
                    }
                    chunks.append(chunk)
                elif num_chars > self.max_chunk_chars:
                    # If the node itself is too large, maybe split further? Or just log.
                    print(
                        f"Warning: Node {getattr(node, 'name', '?')} in {file_path} "
                        f"({num_chars} chars) exceeds max_chunk_chars "
                        f"({self.max_chunk_chars}). Skipping."
                    )

        # Add top-level code as chunks if not part of a larger node already captured
        # (This part needs refinement based on desired chunking strategy for loose code)
        # For now, we focus on functions and classes.

        return chunks


class MarkdownChunker:
    # Use regex to find markdown headers (level, text)
    _MARKDOWN_HEADER = re.compile(r"^(#+)\s+(.+)$")

    def __init__(self, max_header_depth: int = 6):
        # Removed min/max_chunk_chars, added max_header_depth
        self.max_header_depth = max_header_depth
        self._chunk_id_counter = 0  # Internal counter for unique IDs

    @staticmethod
    def _is_md_header(line: str) -> tuple[bool, str, int]:
        """Checks if a line is a markdown header, returns (is_header, text, level)."""
        match = MarkdownChunker._MARKDOWN_HEADER.match(line)
        return (True, match.group(2).strip(), len(match.group(1))) if match else (False, "", 0)

    def _generate_chunk_id(self, base_id: str, header_text: str) -> str:
        """Generates a unique chunk ID."""
        self._chunk_id_counter += 1
        # Sanitize header text for ID
        sanitized_header = re.sub(r"\W+", "_", header_text.lower()).strip("_")
        return f"{base_id}::{self._chunk_id_counter}_{sanitized_header}"

    def _process_section(
        self,
        file_path: str,
        lines: list[str],
        start_index: int,
        parent_chunk_id: str | None = None,  # Parent ID for hierarchy
    ) -> list[dict[str, Any]]:
        """Recursively processes markdown sections based on headers."""
        chunks = []
        is_start_header, start_header_text, start_header_level = self._is_md_header(lines[start_index])

        if not is_start_header or start_header_level > self.max_header_depth:
            return []  # Should start with a valid header within depth limit

        current_chunk_id = self._generate_chunk_id(
            parent_chunk_id or Path(file_path).stem,  # Use file stem if no parent
            start_header_text,
        )

        # --- Scan Phase --- Find children and next sibling/parent
        child_header_indices = []
        processing_end_index = len(lines)  # Where to stop looking overall
        first_child_index = -1  # Index of the first direct child found

        in_code_block_scan = False
        for i in range(start_index + 1, len(lines)):
            line = lines[i]
            if line.strip().startswith("```"):
                in_code_block_scan = not in_code_block_scan
            if in_code_block_scan:
                continue

            is_header, _, header_level = self._is_md_header(line)
            if is_header:
                if header_level <= start_header_level:
                    processing_end_index = i  # Found sibling/parent, stop scan
                    break
                elif header_level == start_header_level + 1 and header_level <= self.max_header_depth:
                    child_header_indices.append(i)
                    if first_child_index == -1:
                        first_child_index = i  # Record the first child index
        # else: Loop completed without break, processing_end_index remains len(lines)

        # --- Content Slice Phase ---
        # Content ends either before the first child or before the next sibling/parent
        content_end_index = processing_end_index
        if first_child_index != -1:
            # If a child was found, content ends before it, even if processing continues further
            content_end_index = first_child_index

        content_lines = lines[start_index:content_end_index]
        current_chunk_content = "\n".join(content_lines).strip()

        # --- Chunk Creation ---
        # Only add chunk if it has meaningful content beyond just the header
        if current_chunk_content and current_chunk_content != lines[start_index]:
            chunks.append(
                {
                    "chunk_id": current_chunk_id,
                    "file_path": file_path,
                    "start_line": start_index + 1,
                    "end_line": content_end_index,
                    "content": current_chunk_content,
                    "type": "MarkdownHeaderChunk",
                    "name": start_header_text,
                    "header": start_header_text,
                    "level": start_header_level,
                    "parent_chunk_id": parent_chunk_id,
                }
            )

        # --- Child Processing Phase ---
        for child_index in child_header_indices:
            # Process all children identified during the scan phase
            child_chunks = self._process_section(file_path, lines, child_index, current_chunk_id)
            chunks.extend(child_chunks)

        # No adjustment logic needed
        return chunks

    def chunk_file(self, file_path: Path) -> list[dict[str, Any]]:
        content = read_file_content(file_path)
        if not content:
            return []

        lines = content.splitlines()
        all_chunks = []
        self._chunk_id_counter = 0  # Reset counter for each file

        processed_until = 0
        # Process top-level sections
        for i, line in enumerate(lines):
            if i < processed_until:
                continue  # Skip lines already processed by a previous section

            is_header, _, level = self._is_md_header(line)
            if is_header and level <= self.max_header_depth:
                if section_chunks := self._process_section(
                    str(file_path),
                    lines,
                    i,
                    parent_chunk_id=None,  # Top-level sections have no parent ID
                ):
                    all_chunks.extend(section_chunks)
                    # Update processed_until based on the end of the last chunk from this section
                    # Need to find the max end_line across all chunks originating from this top-level header
                    max_end_line = 0
                    q = list(section_chunks)
                    while q:
                        chunk = q.pop(0)
                        max_end_line = max(max_end_line, chunk["end_line"])
                    last_chunk_end = section_chunks[-1].get("end_line", i + 1) if section_chunks else i + 1
                    processed_until = max(processed_until, last_chunk_end)  # Ensure progress

        # TODO: Optionally handle content before the first header as a separate chunk if needed.
        # Current logic only chunks starting from the first valid header.

        return all_chunks
