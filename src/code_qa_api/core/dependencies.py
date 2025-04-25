from code_qa_api.core.config import settings
from code_qa_api.rag.store import VectorStore

_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Dependency function to get the singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        if settings.vector_store_path is None:
            raise ValueError("vector_store_path must be set in settings")
        if not settings.vector_store_path.parent.exists():
            settings.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {settings.vector_store_path.parent}")
        _vector_store_instance = VectorStore(persist_directory=settings.vector_store_path)
        print(f"Initialized VectorStore with path: {settings.vector_store_path}")
    return _vector_store_instance
