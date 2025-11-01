#!/usr/bin/env python3
"""
Modal.com Inference Server with Custom LoRA Model

Deploy with: modal deploy modal_inference_server_v2.py

Access the API:
  Base URL: https://hsomex0000--soyebot-llm-v2-api.modal.run
  Health: GET /
  Chat: POST /v1/chat/completions (OpenAI-compatible)
"""

import time
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from modal import App, Image, Volume, asgi_app

# ============================================================================
# Configuration
# ============================================================================

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER_PATH = "/model_vol/soyemodel"

# ============================================================================
# Create Modal App with Image and Volume
# ============================================================================

image = Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "pydantic",
    "fastapi",
)

app = App(name="soyebot-llm-v2", image=image)

# Mount the volume containing the custom LoRA model
model_volume = Volume.from_name("soyebot-model-volume")

# ============================================================================
# Pydantic Models
# ============================================================================


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "soyebot-custom-model"
    messages: List[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = 128


# ============================================================================
# LLM Class with LoRA Support
# ============================================================================


class LLMInference:
    """GPU-backed LLM with LoRA adapter."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """Load base model and LoRA adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Loading base model {BASE_MODEL} on {self.device}...")

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else self.device,
            trust_remote_code=True,
        )

        # Load and merge LoRA adapter
        print(f"🎯 Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
        self.model = PeftModel.from_pretrained(self.model, LORA_ADAPTER_PATH)

        # Merge LoRA weights into base model for inference speedup
        print("⚡ Merging LoRA weights into base model...")
        self.model = self.model.merge_and_unload()

        self.model.eval()
        print("✅ Model and LoRA adapter loaded successfully")

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
    return {
        "status": "ok",
        "model": "soyebot-custom-model",
        "base_model": BASE_MODEL,
        "lora_adapter": LORA_ADAPTER_PATH,
    }


def _chat_completions_handler(request: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat completion endpoint handler."""
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


@web_app.post("/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat completion endpoint (without /v1 prefix)."""
    return _chat_completions_handler(request)


@web_app.post("/v1/chat/completions")
def chat_completions_v1(request: ChatCompletionRequest) -> dict:
    """OpenAI-compatible chat completion endpoint (with /v1 prefix)."""
    return _chat_completions_handler(request)


# ============================================================================
# Mount FastAPI app to Modal with Volume
# ============================================================================

@app.function(volumes={"/model_vol": model_volume})
@asgi_app()
def api():
    """Modal web app handler with access to model volume."""
    return web_app
