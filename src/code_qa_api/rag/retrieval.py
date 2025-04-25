from typing import Any

from code_qa_api.rag.embedding import get_embeddings
from code_qa_api.rag.store import VectorStore


async def retrieve_relevant_chunks(question: str, vector_store: VectorStore, k: int = 5) -> Any:
    if not vector_store.is_initialized():
        print("Warning: Vector store is not initialized or empty. Cannot retrieve.")
        return []

    # Get embedding for the question
    question_embedding = await get_embeddings([question])
    if question_embedding.size == 0:
        print("Could not generate embedding for the question.")
        return []

    # Search the vector store
    try:
        results = vector_store.search(question_embedding[0], k=k)
        # Results are dictionaries containing metadata
        print(f"Retrieved {len(results)} chunks for the question.")
        return results
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []
