"""Internal LLM service for SoyeBot using HuggingFace Transformers."""

import asyncio
import logging
import re
from typing import Optional, Tuple, Any, List, Dict
from threading import Thread
from queue import Queue

import torch
import discord
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT

logger = logging.getLogger(__name__)


class InternalLLMService:
    """Internal LLM service using HuggingFace Transformers with GPU acceleration."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()

    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.config.model_path}...")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU (this will be slow)")

        try:
            # Check if this is a LoRA adapter model
            import os
            is_lora = os.path.exists(os.path.join(self.config.model_path, "adapter_config.json"))

            if is_lora:
                logger.info("Detected LoRA adapter model. Loading base model and adapter...")

                # Load PEFT config to get base model name
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                base_model_name = peft_config.base_model_name_or_path

                logger.info(f"Base model: {base_model_name}")

                # Load tokenizer from adapter path (it should have the tokenizer files)
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                )

                # Load base model
                logger.info("Loading base model...")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )

                # Load LoRA adapter
                logger.info("Loading LoRA adapter...")
                self.model = PeftModel.from_pretrained(
                    base_model,
                    self.config.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                )
            else:
                logger.info("Loading full fine-tuned model...")

                # Load tokenizer
                logger.info("Loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=True
                )

                # Load model with optimizations
                logger.info("Loading model weights...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )

                if self.device == "cpu":
                    self.model = self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode

            logger.info("✅ Model loaded successfully!")

            # Log model info
            if self.device == "cuda":
                logger.info(f"Model memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _generate_text(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> str:
        """Generate text from messages using the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Generated text
        """
        try:
            # Format messages for the model
            # Qwen models typically use ChatML format
            formatted_prompt = self._format_messages(messages)

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,  # Adjust based on model's context length
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Qwen model (ChatML format).

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        # Qwen models use ChatML format
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # Add assistant prompt for generation
        formatted += "<|im_start|>assistant\n"

        return formatted

    async def _generate_async(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Async wrapper for text generation.

        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_text(messages, max_new_tokens, temperature)
        )

    async def summarize_text(self, text: str) -> Optional[str]:
        """Summarize text using the internal model.

        Args:
            text: Text to summarize

        Returns:
            Summary text or None if failed
        """
        if not text.strip():
            logger.debug("Summarization requested for empty text")
            return "요약할 메시지가 없습니다."

        logger.info(f"Summarizing text ({len(text)} characters)...")

        prompt = f"Discord 대화 내용:\n{text}"
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._generate_async(messages, max_new_tokens=self.config.summary_max_tokens, temperature=0.7)
            return response
        except Exception as e:
            logger.error(f"Summarization failed: {e}", exc_info=True)
            return None

    async def generate_chat_response(
        self,
        chat_session: "InternalLLMChatSession",
        user_message: str,
        discord_message: discord.Message,
        tools: Optional[list] = None,
    ) -> Optional[Tuple[str, Optional[Any]]]:
        """Generate chat response with conversation history.

        Args:
            chat_session: InternalLLMChatSession object
            user_message: User message
            discord_message: Discord message object
            tools: Optional list of function calling tools (not supported)

        Returns:
            Tuple of (response_text, None) or None
        """
        logger.debug(
            f"Generating chat response - User message length: {len(user_message)}"
        )

        try:
            # Add user message to history
            chat_session.add_user_message(user_message)
            messages = chat_session.get_messages_with_system_prompt()

            # Generate response
            response_text = await self._generate_async(
                messages,
                max_new_tokens=self.config.max_tokens,
                temperature=0.7
            )

            if not response_text:
                logger.error("Empty response from model")
                chat_session.messages.pop()  # Remove the user message we added
                return None

            # Add assistant response to history
            chat_session.add_assistant_message(response_text)

            return (response_text, None)

        except Exception as e:
            logger.error(f"Chat response generation failed: {e}", exc_info=True)
            # Remove the message we added since generation failed
            if chat_session.messages and chat_session.messages[-1]["role"] == "user":
                chat_session.messages.pop()

            await discord_message.reply(
                "❌ 응답 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                mention_author=False,
            )
            return None

    def parse_function_calls(self, response_obj) -> list:
        """Parse function calls from response (not supported for internal model).

        Args:
            response_obj: Response object

        Returns:
            Empty list
        """
        return []


class InternalLLMChatSession:
    """Manages chat session with conversation history for internal LLM."""

    def __init__(self, system_instruction: str):
        """Initialize chat session.

        Args:
            system_instruction: System prompt for the chat
        """
        self.system_instruction = system_instruction
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str) -> None:
        """Add user message to history.

        Args:
            content: User message content
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to history.

        Args:
            content: Assistant message content
        """
        self.messages.append({"role": "assistant", "content": content})

    def get_messages_with_system_prompt(self) -> List[Dict[str, str]]:
        """Get messages with system prompt prepended.

        Returns:
            List of messages with system prompt at the beginning
        """
        return [
            {"role": "system", "content": self.system_instruction},
        ] + self.messages

    def clear(self) -> None:
        """Clear chat history."""
        self.messages = []
