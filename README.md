# Code QA API

A REST API using a Retrieval-Augmented Generation (RAG) system to answer natural language questions about a local code repository.

## Features

- **FastAPI Backend**: Modern, fast web framework.
- **RAG Pipeline**: Extracts code, chunks it logically (functions/classes), generates explanations and embeddings, stores them, retrieves relevant chunks, and generates answers using an LLM.
- **`litellm` Integration**: Supports various LLM providers.
- **AST-based Chunking**: Creates meaningful code chunks.
- **Vector Storage**: Uses FAISS for efficient similarity search.
- **Configurable**: Settings managed via environment variables.
- **Automated Evaluation**: Script to measure answer quality against a reference dataset.
- **Swagger UI**: Interactive API documentation at `/`.

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
    - `LLM_MODEL`: The identifier for the LLM you want `litellm` to use (e.g., `openai/gpt-4o`, `claude-3-opus-20240229`).
    - `EMBEDDING_MODEL`: The identifier for the embedding model (e.g., `openai/text-embedding-ada-002` or a sentence-transformer model like `all-MiniLM-L6-v2`). LiteLLM can proxy embedding calls too.
    - `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.: API keys for your chosen LLM provider.
    - `REPO_PATH`: **Required for indexing on startup.** The absolute path to the local code repository you want to query.
    - `INDEX_PATH`: Path to store the FAISS index and metadata (defaults to `data/index`).
    - `QA_DATA_PATH`: Path to the cloned `grip_qa` repository or the YAML file for evaluation (defaults to `data/grip_qa/qa.yaml`).
    - `CHUNK_EXPLANATION_MODEL`: LLM model used specifically for generating chunk explanations during indexing.

5.  **Download Evaluation Data (Optional but recommended):**
    Clone the `grip_qa` repository into the `data/` directory:
    ```bash
    git clone https://github.com/Modelcode-ai/grip_qa.git data/grip_qa
    ```
    Or manually download the `qa.yaml` file into `data/grip_qa/`.

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
        "prompt_template": "Answer the question '{question}' based only on the following code context:\n{context}"
      }
      ```
      (`prompt_template` is optional)
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
