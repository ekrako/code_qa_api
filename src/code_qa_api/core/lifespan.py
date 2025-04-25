from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from code_qa_api.core.config import settings
from code_qa_api.core.dependencies import get_vector_store  # Import the dependency function
from code_qa_api.rag.indexing import index_repository


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load the ML model
    print("Lifespan start: Initializing resources...")
    vector_store = get_vector_store()
    # Trigger indexing if REPO_PATH is set
    repo_path_str = settings.repo_path
    if not repo_path_str:
        print("REPO_PATH environment variable not set. No initial indexing will occur.")
        print("Set the REPO_PATH environment variable to index a repository on startup.")
        raise ValueError("REPO_PATH environment variable not set. No initial indexing will occur.")
    if repo_path_str == "must_provide_repo_path":
        raise ValueError("REPO_PATH environment variable not set. No initial indexing will occur.")
    repo_path = Path(repo_path_str)
    if not repo_path.is_dir():
        raise ValueError(f"REPO_PATH '{repo_path_str}' is set but is not a valid directory. Skipping initial indexing.")

    print(f"REPO_PATH set to '{repo_path}'. Starting initial indexing...")
    await index_repository(repo_path, vector_store, force_overwrite=settings.force_overwrite)
    print(f"Initial indexing finished for {repo_path}.")

    yield  # Application runs here

    # Clean up the ML models and release the resources
    print("Lifespan end: Cleaning up resources...")
    # vector_store.save_index() # Chroma persists automatically, explicit save might not be needed or desired here
