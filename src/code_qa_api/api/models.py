from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(..., description="The natural language question about the codebase.")


class QAResponse(BaseModel):
    answer: str = Field(..., description="The generated answer to the question.")


class IndexRequest(BaseModel):
    repo_path: str = Field(..., description="The absolute path to the local code repository to index.")


class IndexResponse(BaseModel):
    message: str = Field(..., description="Status message indicating the result of the indexing process.")
    indexed_files: int = Field(..., description="Number of files successfully indexed.")
