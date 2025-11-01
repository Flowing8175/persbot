#!/usr/bin/env python3
"""
Modal.com Inference Server with OpenAI-compatible API

Deploy with: modal deploy modal_inference_server_v2.py

Access the API:
  Base URL: https://hsomex0000--soyebot-llm-v2.modal.run
  Health: GET /
  Chat: POST /v1/chat/completions (OpenAI-compatible)
"""

import time
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from modal import App, Image, asgi_app

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# ============================================================================
# Create Modal App with Image
# ============================================================================

image = Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "pydantic",
    "fastapi",
)

app = App(name="soyebot-llm-v2", image=image)

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
# LLM Class (GPU-backed)
# ============================================================================


class LLMInference:
    """GPU-backed LLM inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """Load model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Loading {MODEL_NAME} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        print("✅ Model loaded")

    def generate(self, prompt: str, temp: float = 0.7, top_p: float = 0.9, max_tokens: int = 128) -> str:
        """Generate response from prompt."""
        import torch

        if not self.model:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_tokens + inputs["input_ids"].shape[1],
                temperature=temp,
                top_p=top_p,
                do_sample=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================================
# FastAPI App with Web Endpoints
# ============================================================================


web_app = FastAPI()


@web_app.get("/")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_NAME}


@web_app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat completion endpoint."""
    try:
        llm = LLMInference()
        llm.load()

        # Build prompt from messages
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"

        # Generate response
        response_text = llm.generate(
            prompt=prompt,
            temp=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 128,
        )

        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text.strip()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split()),
            },
        }

    except Exception as e:
        return {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }


# ============================================================================
# Mount FastAPI app to Modal
# ============================================================================

@app.function()
@asgi_app()
def api():
    """Modal web app handler."""
    return web_app
