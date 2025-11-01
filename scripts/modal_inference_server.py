#!/usr/bin/env python3
"""
Modal.com Inference Server with OpenAI-compatible API.

Deploy with: modal deploy modal_inference_server.py

Access the API at: https://workspace--soyebot-llm.modal.run/v1
"""

import os
import sys
import json
import time
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from modal import App, Image, Mount, web_endpoint
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread


# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = "/models/merged_model"  # Will be mounted from your local disk
LORA_PATH = "/models/lora_adapter"  # Alternative: use LoRA directly

# Use merged model if it exists, fallback to LoRA + base model
USE_MERGED_MODEL = True

# Model config
MAX_LENGTH = 2048
DEVICE = "cuda"  # Modal provides NVIDIA GPUs


# ============================================================================
# Modal App Setup
# ============================================================================

# Create container image with dependencies
image = Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
)

app = App("soyebot-llm", image=image)


# ============================================================================
# OpenAI-Compatible Request/Response Models
# ============================================================================


class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "soyebot-model"
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = 128
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


# ============================================================================
# Model Loading (runs at container startup)
# ============================================================================


class LLMModel:
    """Wrapper for model and tokenizer."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE

    def load(self):
        """Load model and tokenizer."""
        print(f"🤖 Loading model from {MODEL_PATH}...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                use_fast=True,
            )
            print("✅ Tokenizer loaded")

            # Load model
            if torch.cuda.is_available():
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    load_in_4bit=False,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    torch_dtype=torch.float32,
                    device_map=self.device,
                    trust_remote_code=True,
                )

            self.model.eval()
            print("✅ Model loaded and ready")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 128,
        stream: bool = False,
    ) -> str:
        """Generate response from prompt."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if stream:
                # For streaming
                streamer = TextIteratorStreamer(
                    self.tokenizer, skip_special_tokens=True
                )
                generation_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "streamer": streamer,
                    "max_length": max_tokens + inputs["input_ids"].shape[1],
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                }
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                output = ""
                for text in streamer:
                    output += text
                    yield text
                return
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_tokens + inputs["input_ids"].shape[1],
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Initialize global model instance
_model = None


def get_model() -> LLMModel:
    """Get or create model instance."""
    global _model
    if _model is None:
        _model = LLMModel()
        _model.load()
    return _model


# ============================================================================
# Modal Functions
# ============================================================================


@app.cls(timeout=600)
class LLMServer:
    """Modal GPU-backed LLM server."""

    def __enter__(self):
        """Load model on container startup."""
        print("🚀 Initializing LLM server...")
        self.model = LLMModel()
        self.model.load()

    @web_endpoint(
        method="POST",
        docs=False,
    )
    async def chat_completions(self, request: dict) -> dict:
        """
        OpenAI-compatible chat completions endpoint.

        Accepts: POST /v1/chat/completions
        """
        try:
            req = ChatCompletionRequest(**request)

            # Convert messages to prompt (simple format for now)
            prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in req.messages])
            prompt += "\nassistant:"

            # Generate response
            response_text = self.model.generate(
                prompt=prompt,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens or 128,
                stream=req.stream,
            )

            # Return OpenAI-compatible response
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text.strip(),
                        },
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
            print(f"❌ Error in chat_completions: {e}")
            return {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            }

    @web_endpoint(method="GET", docs=True)
    async def health(self) -> dict:
        """Health check endpoint."""
        return {"status": "ok", "model": "soyebot-qwen3-4b"}


# ============================================================================
# Local Testing
# ============================================================================


@app.local_entrypoint()
def main():
    """Local testing mode (for development)."""
    print("🧪 Testing in local mode...")

    server = LLMServer()

    # Test request
    test_request = {
        "model": "soyebot-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
        "temperature": 0.7,
        "max_tokens": 128,
    }

    response = server.chat_completions(test_request)
    print("\n📝 Response:")
    print(json.dumps(response, indent=2))
