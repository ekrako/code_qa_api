from pathlib import Path
from typing import Any, Mapping, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from code_qa_api.core.config import settings  # noqa: E402

PrimitiveData = str | int | float | bool  # Define alias if not importable


# ChromaDB based vector store


class VectorStore:
    def __init__(
        self,
        collection_name: str = "code_qa_collection",
        persist_directory: Optional[Path] = settings.vector_store_path,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        if self.persist_directory:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            print(f"Initialized persistent ChromaDB client at {self.persist_directory}")
        else:
            # In-memory client
            self._client = chromadb.Client(settings=ChromaSettings(anonymized_telemetry=False))
            print("Initialized in-memory ChromaDB client")

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # Consider adding metadata for embedding function if using Chroma's default
            # metadata={"hnsw:space": "l2"} # Default is l2, matching Faiss IndexFlatL2
        )
        print(f"Got or created ChromaDB collection: '{self.collection_name}'")

    def add(self, embeddings: np.ndarray, metadatas: list[dict[str, Any]]) -> None:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)  # Reshape for single embedding
        if not isinstance(metadatas, list):
            metadatas = [metadatas]  # Ensure metadatas is a list

        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadatas")

        # ChromaDB expects lists for embeddings and metadatas
        embeddings_list = embeddings.tolist()

        # Generate unique IDs for each chunk
        # Using file_path and chunk_id from metadata seems robust
        ids = [f"{meta.get('file_path', 'unknown')}_{meta.get('chunk_id', idx)}" for idx, meta in enumerate(metadatas)]

        # Check for duplicate IDs before adding
        existing_items = self._collection.get(ids=ids)
        ids_to_add = []
        embeddings_to_add = []
        metadatas_to_add = []

        if existing_items and existing_items["ids"]:
            existing_ids_set = set(existing_items["ids"])
            for i, item_id in enumerate(ids):
                if item_id not in existing_ids_set:
                    ids_to_add.append(item_id)
                    embeddings_to_add.append(embeddings_list[i])
                    metadatas_to_add.append(metadatas[i])
        else:
            # No existing items found for these IDs, add all
            ids_to_add = ids
            embeddings_to_add = embeddings_list
            metadatas_to_add = metadatas

        if not ids_to_add:
            # print("No new items to add.") # Optional: log if nothing new
            return  # Nothing to add

        # Ignore type error: Assume runtime data conforms to ChromaDB's expected PrimitiveData
        self._collection.add(embeddings=embeddings_to_add, metadatas=metadatas_to_add, ids=ids_to_add)  # type: ignore[arg-type]

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[Optional[Mapping[str, PrimitiveData]]]:
        count = self._collection.count()
        if count == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)  # Reshape for single query

        # Ensure k is not greater than the number of items in the collection
        k = min(k, count)

        # ChromaDB expects a list of embeddings for querying
        query_embeddings_list = query_embedding.tolist()

        results = self._collection.query(
            query_embeddings=query_embeddings_list,
            n_results=k,
            include=["metadatas"],  # Only need metadata for the result format
        )

        # Extract metadatas from the results
        # Results format: {'ids': [[..]], 'embeddings': None, 'documents': None,
        # 'metadatas': [[..]], 'distances': [[..]]}
        # We are interested in the first list of metadatas as we query
        # with one embedding
        metadatas_result = results.get("metadatas")
        # Ensure metadatas_result is not None and has at least one list
        if metadatas_result and len(metadatas_result) > 0:
            # Mypy struggles with complex optional nested lists from get
            metadatas = metadatas_result[0]  # type: ignore[index]
            # The type of metadatas here is list[Optional[Mapping[str, PrimitiveData]]]
            # which matches the return type hint.
            return metadatas if metadatas is not None else []  # type: ignore[return-value]
        else:
            return []  # Return empty list if metadatas is None or empty

    def reset(self) -> None:
        try:
            self._client.delete_collection(name=self.collection_name)
            print(f"Deleted ChromaDB collection: '{self.collection_name}'")
            # Recreate the collection immediately after deletion
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            print(f"Recreated ChromaDB collection: '{self.collection_name}'")
        except Exception as e:
            # Catch potential errors, e.g., collection doesn't exist
            print(f"Error resetting collection '{self.collection_name}': {e}")
            # Attempt to create it just in case it didn't exist
            try:
                self._collection = self._client.get_or_create_collection(name=self.collection_name)
                print(f"Ensured ChromaDB collection '{self.collection_name}' exists.")
            except Exception as create_e:
                print(f"Failed to ensure collection '{self.collection_name}' exists after reset attempt: {create_e}")

    def is_initialized(self) -> bool:
        # Check if the collection exists and has items
        try:
            # Check if collection exists by trying to get it (raises exception if not)
            # self._client.get_collection(name=self.collection_name) # Alternative check
            # A more direct check is the count
            return self._collection.count() > 0
        except Exception:
            # If get_collection throws or count fails, assume not initialized properly
            return False
