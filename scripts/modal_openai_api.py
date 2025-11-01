#!/usr/bin/env python3
"""
Modal API Wrapper providing OpenAI-compatible /v1/chat/completions endpoint.

This wraps the modal_inference_server_v2 endpoints to provide proper OpenAI format.

Deploy with: modal deploy modal_openai_api.py

Access the API at: https://hsomex0000--soyebot-llm-openai.modal.run/v1/chat/completions
"""

from typing import List, Optional
from pydantic import BaseModel
from modal import App, Image, fastapi_endpoint
from fastapi import FastAPI

# ============================================================================
# Configuration
# ============================================================================

# Modal endpoint for chat inference
MODAL_CHAT_ENDPOINT = "https://hsomex0000--soyebot-llm-v2-chat-inference.modal.run"

# ============================================================================
# Create App
# ============================================================================

image = Image.debian_slim(python_version="3.10").pip_install(
    "fastapi",
    "httpx",
    "pydantic",
    "uvicorn",
)

app = App(name="soyebot-llm-openai", image=image)

# ============================================================================
# Pydantic Models
# ============================================================================


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "soyebot-model"
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = 128


# ============================================================================
# Web Endpoints
# ============================================================================


@app.function()
@fastapi_endpoint(method="GET")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}


@app.function()
@fastapi_endpoint(method="POST")
async def chat_completions(request: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat completions endpoint."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                MODAL_CHAT_ENDPOINT,
                json=request.model_dump(),
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }
