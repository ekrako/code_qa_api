import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from code_qa_api.core.config import settings
from code_qa_api.main import app
from code_qa_api.rag.store import VectorStore

# Ensure test environment uses a temporary index
TEST_INDEX_PATH = Path("./test_data/index")

# Example Python code to be used in tests
SAMPLE_PYTHON_CODE = '''
import os

class MyClass:
    def __init__(self, name: str):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}!"
def top_level_function(x, y):
    """This is a docstring."""
    z = x + y  # Calculation
    print(f"Result: {z}")
    return z
'''


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # Create test directories
    TEST_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    yield  # Run tests
    # Teardown: remove test data
    shutil.rmtree(TEST_INDEX_PATH.parent)

@pytest.fixture(scope="session")
def event_loop():
    # Override default pytest-asyncio event loop to match FastAPI's needs if necessary
    # Usually not needed unless encountering loop issues
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    # Use TestClient for synchronous testing or within async tests
    with TestClient(app) as test_client:
        yield test_client


# Fixture for a clean VectorStore instance per test function
@pytest.fixture(scope="function")
def vector_store() -> Generator[VectorStore, None, None]:
    # Use a unique temporary directory for each test run
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = Path(tmpdir) / "chroma_test_index"
        # Ensure the path exists (though PersistentClient might do this)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Initialize the store within the temporary directory
        store = VectorStore(persist_directory=persist_path)

        yield store

        # Teardown: Reset store and the temporary directory is removed automatically
        try:
            store.reset()
        except Exception as e:
            print(f"Error resetting vector store during teardown: {e}")

        # No need for manual rmtree, TemporaryDirectory handles it
        # print(f"Temporary directory {tmpdir} will be cleaned up.")


@pytest.fixture(scope="session")
def test_repo_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()
        (repo_path / "module1.py").write_text(SAMPLE_PYTHON_CODE)
        (repo_path / "subdir").mkdir()
        (repo_path / "subdir" / "module2.py").write_text("def another_func(): pass")
        # Set environment variable for lifespan initialization test
        original_repo_path = os.environ.get("REPO_PATH")
        os.environ["REPO_PATH"] = str(repo_path.resolve())
        yield repo_path.resolve()
        # Clean up environment variable
        if original_repo_path is None:
            del os.environ["REPO_PATH"]
        else:
            os.environ["REPO_PATH"] = original_repo_path


@pytest.fixture
def mock_settings(test_repo_path: Path, monkeypatch) -> Generator[None, None, None]:
    """Mocks settings, especially the vector store path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store_path = Path(tmpdir) / "test_vector_store"
        # No need to mkdir, PersistentClient does this
        # vector_store_path.mkdir(parents=True, exist_ok=True)

        # Patch the imported settings object
        monkeypatch.setattr(settings, "vector_store_path", vector_store_path)
        monkeypatch.setattr(settings, "repo_path", test_repo_path)

        yield
