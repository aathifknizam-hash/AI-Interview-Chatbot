"""
main.py — FastAPI Gateway
==========================
2026 stack: sentence-transformer backend, Pydantic v2 validators,
async lifespan, structured logging, restricted CORS.

Endpoints
---------
GET  /         → status check
POST /chat     → main inference route
GET  /health   → uptime / readiness probe
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from predict import get_response, load_artifacts

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_coach.main")

# ── Constants ──────────────────────────────────────────────────────────────
APP_VERSION = "2.0.0"

# In production set  ALLOWED_ORIGINS="https://yourfrontend.com"  as an env var.
# Falls back to localhost Streamlit port for local development.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting — loading semantic index …")
    try:
        load_artifacts()
        logger.info("Semantic index loaded. Server is ready.")
    except FileNotFoundError as exc:
        # Log clearly but don't crash the process; /health will report unhealthy
        logger.error("Could not load artifacts on startup: %s", exc)
        logger.error("Run train.py first to build the semantic index.")
    yield
    logger.info("Server shutting down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Interview Coach API",
    description=(
        "Semantic intent classification powered by sentence-transformers. "
        "Run train.py once to build the index, then start the server."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,           # credentials not required; keeps CSP simple
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again."},
    )


# ── Schemas ────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

    @field_validator("message", mode="before")
    @classmethod
    def validate_message(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Message must be a string.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace.")
        if len(cleaned) > 600:
            raise ValueError(
                f"Message is too long ({len(cleaned)} characters). "
                "Please keep it under 600 characters."
            )
        return cleaned


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "status": "online",
        "version": APP_VERSION,
        "docs": "/docs",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message and receive an interview coaching response",
    status_code=status.HTTP_200_OK,
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Accepts a user message, runs semantic intent classification, and
    returns the most relevant coaching response from intents.json.
    """
    logger.info("Received message (len=%d)", len(request.message))
    try:
        bot_response = get_response(request.message)
        return ChatResponse(response=bot_response)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model index is not available. Please contact the administrator.",
        )
    except Exception as exc:
        logger.exception("Error in /chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process the message. Please try again.",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Readiness / health probe",
    status_code=status.HTTP_200_OK,
)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", version=APP_VERSION)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # set reload=False in production
        log_level="info",
    )