from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file relative to the script's location or project root
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM Settings
    # generation_model: str = "anthropic/claude-3-7-sonnet-20250219"
    # chunk_explanation_model: str = "anthropic/claude-3-7-sonnet-20250219"
    generation_model: str = "openai/gpt-4.1"
    chunk_explanation_model: str = "openai/gpt-4.1"
    embedding_model: str = "openai/text-embedding-3-small"

    # RAG Settings
    repo_path: str = "/Users/erank/Personal/model_code/grip-no-tests"
    force_overwrite: bool = False  # Whether to overwrite existing index
    index_path: Path = Path("./data/index")

    # Evaluation Settings
    qa_data_path: Path = Path("./data/grip_qa/")
    qa_repo_url: str = "https://github.com/Modelcode-ai/grip_qa.git"
    # API Settings
    api_base_url: str = "http://127.0.0.1:8000"
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")
    api_prefix: str = "/api"
    project_name: str = "Code QA API"
    project_version: str = "0.1.0"
    vector_store_path: Path = Path("./data/chroma_db")
    # Add other settings as needed


settings = Settings()

# Ensure index directory exists
settings.index_path.mkdir(parents=True, exist_ok=True)
