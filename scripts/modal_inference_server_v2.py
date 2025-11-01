#!/usr/bin/env python3
"""
Modal.com Inference Server with OpenAI-compatible API - Simplified Pattern

Deploy with: modal deploy modal_inference_server_v2.py

Access the API at: https://hsomex0000--soyebot-llm-v2.modal.run
"""

import time
import json
from typing import List, Optional

from modal import App, Image, fastapi_endpoint
from pydantic import BaseModel

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
# LLM Class
# ============================================================================


class LLMInference:
    """GPU-backed LLM."""

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
        """Generate response."""
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
# Modal Web Endpoints
# ============================================================================


@app.function()
@fastapi_endpoint(method="GET")
def health_check() -> dict:
    """Health check."""
    return {"status": "ok", "model": MODEL_NAME}


@app.function()
@fastapi_endpoint(method="POST")
def chat_inference(request: ChatCompletionRequest) -> dict:
    """Chat completion endpoint."""
    try:
        llm = LLMInference()
        llm.load()

        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"

        response_text = llm.generate(
            prompt=prompt,
            temp=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 128,
        )

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
