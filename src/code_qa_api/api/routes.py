from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from code_qa_api.api.models import QARequest, QAResponse
from code_qa_api.core.dependencies import get_vector_store
from code_qa_api.rag.generation import generate_answer
from code_qa_api.rag.retrieval import retrieve_relevant_chunks
from code_qa_api.rag.store import VectorStore

router = APIRouter()


_vector_store_instance = None


@router.post("/answer", response_model=QAResponse)
async def answer_question(request: QARequest, vector_store: Annotated[VectorStore, Depends(get_vector_store)]) -> QAResponse:
    if not vector_store.is_initialized():
        raise HTTPException(
            status_code=400,
            detail=("Vector store is not initialized. Indexing might be in progress or REPO_PATH not set."),
        )

    try:
        retrieved_chunks = await retrieve_relevant_chunks(request.question, vector_store, k=5)
        answer = await generate_answer(request.question, retrieved_chunks)
        return QAResponse(answer=answer)
    except Exception as e:
        print(f"Error answering question: {e}")
        # Consider logging the full traceback
        raise HTTPException(status_code=500, detail="Internal server error during question answering.")
