import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from llm_service import InferenceBusyError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai-chatbot-api")

from rag_context import ask_chatbot

# Backend-controlled retrieval configuration
SEMANTIC_WEIGHT = 3.0
KEYWORD_WEIGHT = 1.0


app = FastAPI(
    title="AI Chatbot API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Request-ID"],
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatSource(BaseModel):
    chunk_id: int
    source: str
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)


@app.get("/")
def home():
    return {"message": "AI Chatbot API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, response: Response):

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    response.headers["X-Request-ID"] = request_id

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    logger.info(
        "Chat request received | request_id=%s",
        request_id,
    )

    try:
        result = ask_chatbot(
            question,
            semantic_weight=SEMANTIC_WEIGHT,
            keyword_weight=KEYWORD_WEIGHT,
        )

        return ChatResponse(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
        )

    except InferenceBusyError as error:
        logger.warning("Inference capacity reached | request_id=%s", request_id)

        raise HTTPException(
            status_code=503,
            detail={"message": str(error), "request_id": request_id},
            headers={"Retry-After": "5"},
        ) from error

    except Exception:
        logger.exception(
            "Chat request failed | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "An error occurred while processing your question.",
                "request_id": request_id,
            },
        )

    finally:
        duration = round(
            time.perf_counter() - start_time,
            3,
        )

        logger.info(
            "Chat request finished | request_id=%s | duration=%ss",
            request_id,
            duration,
        )
