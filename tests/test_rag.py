import numpy as np

from code_qa_api.rag.store import VectorStore


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
    assert vector_store.persist_directory is not None
