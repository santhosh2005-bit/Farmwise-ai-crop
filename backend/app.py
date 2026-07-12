"""
app.py — FastAPI entry-point for the FarmWise AI Copilot backend.

Run with:
    uvicorn backend.app:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import config, dataset
from backend.chatbot import Chatbot
from backend.memory import delete_memory

# ─── Logging ─────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)


# ─── Pydantic request / response models ─────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from the frontend."""

    session_id: str = Field(..., description="Unique session identifier.")
    message: str = Field(..., min_length=1, description="User's question.")
    custom_api_key: str | None = Field(None, description="Optional custom user API key.")
    model_choice: str | None = Field(None, description="Optional model choice override.")


class ChatResponse(BaseModel):
    """Structured response returned to the frontend."""

    answer: str = Field(..., description="LLM-generated textual answer.")
    chart_json: str | None = Field(
        None, description="Plotly figure as JSON (None when no chart)."
    )
    charts_json: list[str] | None = Field(
        None, description="List of Plotly charts as JSON."
    )
    data: list[dict[str, Any]] | None = Field(
        None, description="Supporting tabular data."
    )
    suggestions: list[str] | None = Field(
        None, description="Follow-up question suggestions."
    )
    insights: dict[str, Any] | None = Field(
        None, description="Automatic data insights (stats, trends)."
    )
    reasoning: list[dict[str, Any]] | None = Field(
        None, description="Intermediate thoughts and tool calls."
    )


class ResetRequest(BaseModel):
    """Request to reset a conversation session."""

    session_id: str = Field(..., description="Session to reset.")


class DatasetInfoResponse(BaseModel):
    """Metadata about the loaded dataset."""

    columns: list[str]
    row_count: int
    sample_rows: list[dict[str, Any]]
    crops: list[str]
    countries: list[str]
    years: list[int]


# ─── Lifespan: eager-load the dataset once at startup ────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the dataset into memory before the first request."""
    logger.info("Loading dataset...")
    dataset.load_dataset()
    logger.info(f"Dataset loaded: {len(dataset.get_dataframe())} rows")
    yield  # app runs here
    logger.info("Shutting down.")


# ─── App factory ─────────────────────────────────────────────

app = FastAPI(
    title="FarmWise AI Copilot",
    version="0.1.0",
    description="AI-powered agricultural analytics — ask questions in plain English.",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (any origin during development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ──────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health_check() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/dataset/info", response_model=DatasetInfoResponse, tags=["dataset"])
async def dataset_info() -> DatasetInfoResponse:
    """Return metadata about the loaded dataset."""
    df = dataset.get_dataframe()
    return DatasetInfoResponse(
        columns=dataset.get_column_names(),
        row_count=len(df),
        sample_rows=dataset.get_sample_rows(),
        crops=sorted(df["Item"].dropna().unique().tolist()),
        countries=sorted(df["Area"].dropna().unique().tolist()),
        years=sorted(df["Year"].dropna().unique().astype(int).tolist()),
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Accept a user question and return an AI-generated answer.

    Delegates to the ``Chatbot`` orchestrator which manages the
    Groq LLM tool-calling loop, chart generation, and insight
    computation.
    """
    # Validate that the API key is configured
    if not request.custom_api_key and not config.GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured. Please set it in backend/.env or provide a custom key in the settings panel.",
        )

    try:
        bot = Chatbot(
            session_id=request.session_id,
            custom_api_key=request.custom_api_key,
            model_choice=request.model_choice,
        )
        result = await bot.ask(request.message)
    except Exception as e:
        logger.error(f"Chat error for session {request.session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}",
        )

    return ChatResponse(
        answer=result["answer"],
        chart_json=result.get("chart_json"),
        charts_json=result.get("charts_json"),
        data=result.get("data"),
        suggestions=result.get("suggestions"),
        insights=result.get("insights"),
        reasoning=result.get("reasoning"),
    )


@app.post("/chat/reset", tags=["chat"])
async def chat_reset(request: ResetRequest) -> dict[str, str]:
    """Clear conversation memory for a session."""
    delete_memory(request.session_id)
    return {"status": "ok", "message": "Conversation reset."}


@app.get("/analytics/forecast", tags=["analytics"])
async def get_forecast(
    country: str, item: str, forecast_years: int = 5
) -> list[dict[str, Any]]:
    """Retrieve historical and forecasted crop yields."""
    from backend.tools import forecast_yield_trend

    try:
        return forecast_yield_trend(country, item, forecast_years)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/regression", tags=["analytics"])
async def get_regression(country: str, item: str) -> dict[str, Any]:
    """Analyze impact of climate factors and pesticides on crop yield."""
    from backend.tools import get_yield_regression_factors

    try:
        return get_yield_regression_factors(country, item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/raw-data", tags=["analytics"])
async def get_raw_analytics_data(country: str, item: str) -> list[dict[str, Any]]:
    """Retrieve raw historical yield and climate records."""
    from backend.tools import get_yield_trend

    try:
        return get_yield_trend(country, item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Stand-alone execution ───────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
    )
