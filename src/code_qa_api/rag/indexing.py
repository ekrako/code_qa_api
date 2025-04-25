import time
from pathlib import Path
from typing import Any

from code_qa_api.rag.chunking import MarkdownChunker, PythonCodeChunker
from code_qa_api.rag.embedding import process_chunks_for_embedding
from code_qa_api.rag.store import VectorStore
from code_qa_api.utils.file_handler import find_markdown_files, find_python_files


async def index_repository(
    repo_path: Path,
    vector_store: VectorStore,
    force_overwrite: bool = False,
    batch_size: int = 32,
) -> int:
    print(f"Starting indexing for repository: {repo_path}, Force Overwrite: {force_overwrite}")
    if force_overwrite:
        print("Force overwrite enabled. Clearing existing index...")
        vector_store.reset()
    elif vector_store.is_initialized() and vector_store._collection.count() > 0:
        print("Index already exists and force_overwrite is False. Skipping indexing.")
        return int(vector_store._collection.count())
    else:  # If not initialized or empty, reset just in case
        vector_store.reset()

    start_time = time.time()

    py_files_list = list(find_python_files(repo_path))
    md_files_list = list(find_markdown_files(repo_path))
    all_files_list = py_files_list + md_files_list

    print(f"Found {len(py_files_list)} Python files and {len(md_files_list)} Markdown files to index.")

    all_chunks: list[dict[str, Any]] = []
    processed_files_count = 0
    python_chunker = PythonCodeChunker()
    markdown_chunker = MarkdownChunker()

    # 1. Chunk all files
    for file_path in all_files_list:
        try:
            file_chunks = []
            if file_path.suffix == ".py":
                file_chunks = python_chunker.chunk_file(file_path)
            elif file_path.suffix.lower() == ".md":
                file_chunks = markdown_chunker.chunk_file(file_path)

            if file_chunks:
                all_chunks.extend(file_chunks)
                processed_files_count += 1
        except Exception as e:
            print(f"Error chunking file {file_path}: {e}. Skipping.")

    print(f"Generated {len(all_chunks)} chunks from {processed_files_count} files.")

    if not all_chunks:
        print("No chunks were generated. Indexing complete (0 chunks added).")
        return 0

    # 2. Process chunks in batches (embedding + metadata preparation)
    total_chunks_processed = 0
    if all_chunks:
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} of {len(all_chunks) // batch_size + 1} with {len(batch_chunks)} chunks...")
            try:
                embeddings, metadata = await process_chunks_for_embedding(batch_chunks)
                if embeddings.size > 0:
                    vector_store.add(embeddings, metadata)
                    total_chunks_processed += len(metadata)
                else:
                    print(f"Warning: Batch {i // batch_size + 1} resulted in no embeddings.")
            except Exception as e:
                print(f"Error processing batch {i // batch_size + 1}: {e}. Skipping batch.")
                # Potentially add more robust error handling/retry for batches

    # 3. Finalize
    end_time = time.time()
    duration = end_time - start_time

    final_index_size = vector_store._collection.count()
    print(
        f"Indexing complete in {duration:.2f} seconds. Added {total_chunks_processed} new chunks."
        f" Final index size: {final_index_size} chunks from {processed_files_count} files."
    )
    return int(final_index_size)
