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
import modal

# ============================================================================
# Configuration
# ============================================================================

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_ADAPTER_PATH = "/model_vol/soyemodel"

# ============================================================================
# Create Modal App with Image and Volume
# ============================================================================

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "pydantic",
    "fastapi",
)

app = modal.App(name="soyebot-llm-v2", image=image)

# Mount the volume containing the custom LoRA model
model_volume = modal.Volume.from_name("soyebot-model-volume")

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
# Modal Class with Persistent Model Loading
# ============================================================================


@app.cls(
    gpu="T4",  # Free-tier eligible GPU
    memory=12288,  # 12GB RAM for model inference
    volumes={"/model_vol": model_volume},
    scaledown_window=300,  # Keep container alive for 5 minutes after last request
    min_containers=1,  # Keep at least 1 container warm (optional, may incur costs)
)
class ModelServer:
    """GPU-backed LLM server with LoRA adapter and persistent model loading."""

    @modal.enter()
    def load_model(self):
        """Load model once when container starts (runs before any requests)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Container starting - loading base model {BASE_MODEL} on {self.device}...")

        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
        print("✅ Container ready - model loaded and cached in memory")

    def generate(self, prompt: str, temp: float = 0.7, top_p: float = 0.9, max_tokens: int = 128) -> str:
        """Generate response from prompt."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_tokens + input_length,
                temperature=temp,
                top_p=top_p,
                do_sample=True,
            )

        # Decode only the newly generated tokens (exclude input prompt)
        generated_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    @modal.asgi_app()
    def api(self):
        """FastAPI web app with OpenAI-compatible endpoints."""
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

        @web_app.post("/chat/completions")
        def chat_completions(request: ChatCompletionRequest) -> dict:
            """OpenAI-compatible chat completion endpoint (without /v1 prefix)."""
            return self._chat_handler(request)

        @web_app.post("/v1/chat/completions")
        def chat_completions_v1(request: ChatCompletionRequest) -> dict:
            """OpenAI-compatible chat completion endpoint (with /v1 prefix)."""
            return self._chat_handler(request)

        return web_app

    def _chat_handler(self, request: ChatCompletionRequest) -> dict:
        """Internal chat completion handler."""
        try:
            # Convert messages to Qwen3 chat template format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # Apply Qwen3 chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response using the loaded model
            response_text = self.generate(
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
