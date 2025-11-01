#!/usr/bin/env python3
"""
Modal.com Inference Server with OpenAI-compatible API.

Deploy with: modal deploy modal_inference_server.py

Access the API at: https://hsomex0000--soyebot-llm.modal.run/chat/completions
"""

import time
from typing import List, Optional

from pydantic import BaseModel

from modal import Image, App, web_endpoint

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
MAX_LENGTH = 2048

# ============================================================================
# Create Modal App
# ============================================================================

image = Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "pydantic",
)

app = App(name="soyebot-llm", image=image)

# ============================================================================
# OpenAI-Compatible Request/Response Models
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
# LLM Inference Class
# ============================================================================


@app.cls(timeout=600)
class LLMInference:
    """GPU-backed LLM inference server."""

    def __enter__(self):
        """Load model on container startup."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Loading model {MODEL_NAME}...")
        print(f"   Device: {self.device}")

        try:
            print("   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
            )

            print("   Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else self.device,
                trust_remote_code=True,
            )

            self.model.eval()
            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 128,
    ) -> str:
        """Generate response from prompt."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_tokens + inputs["input_ids"].shape[1],
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================================
# Web Endpoints
# ============================================================================


@web_endpoint(method="POST", label="chat-completions")
def chat_completions(request: ChatCompletionRequest) -> dict:
    """
    OpenAI-compatible chat completions endpoint.

    Example:
        curl -X POST https://hsomex0000--soyebot-llm.modal.run/chat/completions \
          -H "Content-Type: application/json" \
          -d '{
            "model": "soyebot-model",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
            "max_tokens": 128
          }'
    """
    try:
        llm = LLMInference()

        # Convert messages to prompt
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        prompt += "\nassistant:"

        print(f"📝 Prompt: {prompt[:100]}...")

        # Generate response
        response_text = llm.generate(
            prompt=prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 128,
        )

        print(f"✅ Generated: {response_text[:100]}...")

        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
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
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": {
                "message": str(e),
                "type": "server_error",
            }
        }


@web_endpoint(method="GET", label="health-check")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model": "soyebot-qwen3-4b"}


@web_endpoint(method="GET", label="root")
def root() -> dict:
    """Root endpoint."""
    return {
        "name": "SoyeBot LLM API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat/completions",
        },
    }
