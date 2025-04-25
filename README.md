# Code QA API

This project implements a RESTful API service designed to answer natural language questions about a specific, locally stored code repository. It utilizes a Retrieval-Augmented Generation (RAG) pipeline to provide contextually relevant answers based on the codebase's content. The system first processes the target repository, breaking down the code into logical chunks (like functions or classes) using Abstract Syntax Trees (ASTs). It then generates descriptive explanations and vector embeddings for each chunk. These embeddings are stored in a vector index for efficient similarity searching. When a user poses a question via the API, the system retrieves the most relevant code chunks based on the question's embedding, constructs a context, and feeds this context along with the original question to a Large Language Model (LLM) accessed via `litellm`. The LLM then generates a natural language answer grounded in the provided code context.

## Features

- **FastAPI Backend**: Leverages the high-performance FastAPI framework for building the REST API, offering asynchronous support and automatic interactive documentation.
- **RAG Pipeline**: Implements a full RAG pipeline:
    - **Code Extraction**: Reads files from the specified local repository.
    - **AST-based Chunking**: Intelligently divides code into meaningful units (functions, classes) for better context understanding.
    - **Explanation Generation**: Uses an LLM to generate natural language explanations for each code chunk.
    - **Embedding Generation**: Creates vector embeddings for code chunks and their explanations using configurable embedding models.
    - **Vector Storage**: Employs a vector store for efficient storage and retrieval of code chunk embeddings.
    - **Context Retrieval**: Finds the most relevant code chunks based on the user's question using vector similarity search.
    - **Answer Generation**: Uses a configurable LLM (via `litellm`) to synthesize an answer based on the retrieved context and the user's question.
- **`litellm` Integration**: Provides flexibility by supporting connections to various LLM providers (OpenAI, Anthropic, etc.) for both answer generation and chunk explanation.
- **Vector Storage**: Uses a vector database for efficient similarity search and clustering of dense vectors.
- **Configuration**: Easily configured through environment variables (`.env` file) for specifying LLM models, API keys, repository paths, and index locations.
- **Automated Evaluation**: Includes a script (`scripts/evaluate.py`) to assess the RAG system's performance by comparing generated answers against a predefined question-answer dataset (`grip_qa`).
- **Swagger UI**: Automatically generates interactive API documentation, accessible at the root endpoint (`/`), allowing users to easily test API endpoints.

## Solution Architecture

The system follows a typical Retrieval-Augmented Generation (RAG) pattern:

1.  **Indexing (Offline/Startup)**:
    - The target code repository (specified by `REPO_PATH`) is scanned.
    - Code files are parsed using Abstract Syntax Trees (ASTs).
    - Code is chunked into logical units (functions, classes).
    - An LLM (`CHUNK_EXPLANATION_MODEL`) generates a natural language explanation for each chunk.
    - An embedding model (`EMBEDDING_MODEL`) generates vector embeddings for the code chunk and its explanation.
    - The chunks, explanations, and embeddings are stored in a vector index (`INDEX_PATH`) along with associated metadata.
2.  **Querying (Online/API Request)**:
    - A user sends a natural language question to the `/qa` endpoint.
    - The embedding model generates an embedding for the user's question.
    - The vector index is queried to find the code chunks whose embeddings are most similar to the question embedding.
    - The content of these relevant chunks is retrieved.
    - A prompt is constructed using the user's question and the retrieved code context.
    - This prompt is sent to the primary LLM (`LLM_MODEL`) via `litellm`.
    - The LLM generates an answer based on the provided context.
    - The answer is returned in the API response.

## Main Components

- **`src/`**: Contains the core application code.
    - **`main.py`**: FastAPI application setup, defines API endpoints (`/qa`), and handles application startup logic (including triggering the indexing process).
    - **`config.py`**: Manages application settings using Pydantic, loading values from environment variables.
    - **`rag_pipeline/`**: Houses the logic for the RAG system.
        - **`chunking.py`**: Implements code chunking based on AST parsing.
        - **`embedding.py`**: Handles the generation of vector embeddings.
        - **`indexing.py`**: Manages the creation and loading of the vector index.
        - **`generation.py`**: Contains logic for interacting with the LLM to generate answers and explanations.
        - **`retrieval.py`**: Implements the logic for retrieving relevant chunks from the index.
        - **`schemas.py`**: Defines Pydantic models for data structures used within the RAG pipeline and API.
- **`scripts/`**: Includes utility and evaluation scripts.
    - **`evaluate.py`**: Script for evaluating the API's performance against the `grip_qa` dataset.
- **`tests/`**: Contains unit and integration tests for the application.
- **`data/`**: Default directory for storing the vector index (`data/index/`) and the evaluation dataset (`data/grip_qa/`).
- **`.env`**: Stores environment variables (API keys, paths, model names). Ignored by Git.
- **`.env.example`**: Example environment file template.
- **`requirements.txt`**: Lists Python dependencies.
- **`pyproject.toml`**: Project metadata and build configuration.
- **`README.md`**: This file, providing project documentation.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd code_qa_api
    ```

2.  **Install `uv` (if you don't have it):**
    ```bash
    pip install uv
    # or follow instructions at https://github.com/astral-sh/uv
    ```

3.  **Create a virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    uv pip install -r requirements.txt # Or directly: uv pip install -e '.[dev]' if setup correctly
    uv pip install -e .
    ```

4.  **Configure Environment Variables:**
    Copy `.env.example` to `.env` and fill in the required values:
    ```bash
    cp .env.example .env
    ```
    - `GENERATION_MODEL`: The identifier for the main LLM used for answer generation (e.g., `anthropic/claude-3-haiku-20240307`, `openai/gpt-4o`). Defaults to `anthropic/claude-3-haiku-20240307`.
    - `CHUNK_EXPLANATION_MODEL`: The identifier for the LLM used to generate explanations for code chunks during indexing (e.g., `openai/gpt-4-turbo`). Defaults to `openai/gpt-4-turbo`.
    - `EMBEDDING_MODEL`: The identifier for the embedding model (e.g., `openai/text-embedding-3-small`, `sentence-transformers/all-MiniLM-L6-v2`). Defaults to `openai/text-embedding-3-small`.
    - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.: API keys for your chosen LLM and embedding providers, as required by `litellm`.
    - `REPO_PATH`: **Required.** The absolute path to the local code repository you want to query.
    - `VECTOR_STORE_PATH`: Path to store the ChromaDB vector store (defaults to `data/chroma_db`).
    - `FORCE_OVERWRITE`: Set to `true` to overwrite an existing vector store during indexing (defaults to `false`).
    - `QA_DATA_PATH`: Path to the directory containing the `grip_qa` evaluation data (defaults to `data/grip_qa/`).
    - `QA_REPO_URL`: The URL to clone the `grip_qa` repository if `QA_DATA_PATH` doesn't exist (defaults to `https://github.com/Modelcode-ai/grip_qa.git`).
    - `API_BASE_URL`: Base URL of the running API, used by the evaluation script (defaults to `http://127.0.0.1:8000`).
    - `API_PREFIX`: The prefix for API routes (defaults to `/api`).
    - `PROJECT_NAME`: Name of the project (defaults to `Code QA API`).
    - `PROJECT_VERSION`: Version of the project (defaults to `0.1.0`).

5.  **Download Evaluation Data (Optional but recommended):**
    The application will attempt to clone the repo specified by `QA_REPO_URL` into the `QA_DATA_PATH` if the directory doesn't exist. Alternatively, you can manually clone it:
    ```bash
    git clone https://github.com/Modelcode-ai/grip_qa.git data/grip_qa
    ```

## Running the API

```bash
uvicorn src.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. The Swagger UI documentation is available at the root (`/`).

**Indexing:** Indexing of the repository specified by `REPO_PATH` happens automatically on application startup.

## API Endpoints

- `GET /`: Serves the Swagger UI.
- `POST /qa`: Ask a question about the indexed repository.
    - **Request Body:**
      ```json
      {
        "question": "What does class X do?",
      }
      ```
    - **Response Body:**
      ```json
      {
        "answer": "Class X does..."
      }
      ```

## Running the Evaluation Script

Make sure the API is running and the QA data is available.

```bash
python scripts/evaluate.py
```

This will run the questions from the QA dataset against your running API, compare the results, and print a quality score.

## Running Tests

```bash
pytest tests/
```

## Project Structure

(See `pyproject.toml` and the file tree for details)

## TODO

- Replace placeholder URLs in `pyproject.toml`.
- Implement more sophisticated evaluation metrics.
- Add more robust error handling.
- Consider alternative vector stores.
- Optimize indexing for very large repositories.
