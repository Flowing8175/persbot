#!/usr/bin/env python3
"""
Merge LoRA adapter with base model for Modal deployment.

This script takes the LoRA adapter and merges it with the base model,
creating a standalone merged model that can be easily deployed.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_model(
    lora_model_path: str,
    output_path: str,
    base_model_name: Optional[str] = None,
    device: str = "cpu",
    load_in_8bit: bool = False,
) -> None:
    """
    Merge LoRA adapter with base model.

    Args:
        lora_model_path: Path to LoRA adapter directory
        output_path: Where to save merged model
        base_model_name: Base model name (extracted from adapter_config.json if None)
        device: Device to use ("cuda" or "cpu")
        load_in_8bit: Load base model in 8-bit (saves memory)
    """
    print(f"🔧 Merging LoRA adapter: {lora_model_path}")
    print(f"📁 Output path: {output_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Load adapter config to get base model name if not provided
    if base_model_name is None:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(lora_model_path)
        base_model_name = peft_config.base_model_name_or_path
        print(f"📦 Base model (from config): {base_model_name}")
    else:
        print(f"📦 Base model: {base_model_name}")

    # Load base model
    print("⬇️  Loading base model (this may take a moment)...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else device,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        sys.exit(1)

    print("✅ Base model loaded")

    # Load LoRA adapter
    print("⬇️  Loading LoRA adapter...")
    try:
        model = PeftModel.from_pretrained(base_model, lora_model_path)
    except Exception as e:
        print(f"❌ Failed to load LoRA adapter: {e}")
        sys.exit(1)

    print("✅ LoRA adapter loaded")

    # Merge
    print("🔗 Merging adapter with base model...")
    merged_model = model.merge_and_unload()
    print("✅ Merge complete")

    # Save merged model
    print("💾 Saving merged model...")
    merged_model.save_pretrained(output_path)
    print(f"✅ Merged model saved to: {output_path}")

    # Save tokenizer from LoRA adapter
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print("✅ Tokenizer saved")

    print("\n✨ Merge successful! Model is ready for Modal deployment.")
    print(f"Next: Deploy with modal_inference_server.py using {output_path}")


if __name__ == "__main__":
    # Configuration
    LORA_PATH = "/data/data/com.termux/files/home/soyemodel/soyemodel"
    OUTPUT_PATH = "/tmp/merged_model"  # Change to persistent location if needed
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🚀 Starting LoRA merge on device: {DEVICE}")
    print(f"   GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name()}")

    merge_lora_model(
        lora_model_path=LORA_PATH,
        output_path=OUTPUT_PATH,
        device=DEVICE,
    )
