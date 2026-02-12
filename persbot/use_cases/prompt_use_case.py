"""Prompt use case for handling prompt generation operations.

This use case handles persona creation, prompt generation from concepts,
and prompt management.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from persbot.config import AppConfig
from persbot.constants import META_PROMPT, QUESTION_GENERATION_PROMPT
from persbot.services.llm_service import LLMService
from persbot.services.openai_service import OpenAIService
from persbot.services.zai_service import ZAIService

logger = logging.getLogger(__name__)


@dataclass
class PromptGenerationRequest:
    """Request for generating a system prompt from a concept."""

    concept: str
    questions_and_answers: Optional[str] = None
    use_cache: bool = False


@dataclass
class PromptGenerationResponse:
    """Response from prompt generation."""

    system_prompt: str
    success: bool
    error: Optional[str] = None


@dataclass
class QuestionGenerationRequest:
    """Request for generating clarifying questions."""

    concept: str
    max_questions: int = 5


@dataclass
class QuestionGenerationResponse:
    """Response from question generation."""

    questions: list[dict[str, str]]
    success: bool
    error: Optional[str] = None


@dataclass
class Question:
    """A clarifying question with sample answer."""

    question: str
    sample_answer: str


class PromptUseCase:
    """Use case for handling prompt generation operations."""

    def __init__(
        self,
        config: AppConfig,
        llm_service: LLMService,
    ) -> None:
        """Initialize the prompt use case.

        Args:
            config: Application configuration.
            llm_service: LLM service for prompt generation.
        """
        self.config = config
        self.llm_service = llm_service

    async def generate_prompt_from_concept(
        self, request: PromptGenerationRequest
    ) -> PromptGenerationResponse:
        """Generate a detailed system prompt from a simple concept.

        Args:
            request: The prompt generation request.

        Returns:
            PromptGenerationResponse with generated prompt.

        Raises:
            PromptGenerationException: If generation fails.
        """
        try:
            if request.questions_and_answers:
                return await self._generate_prompt_with_answers(request)

            return await self._generate_prompt_from_concept(request)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}", exc_info=True)
            return PromptGenerationResponse(
                success=False,
                system_prompt="",
                error=str(e),
            )

    async def generate_questions(
        self, request: QuestionGenerationRequest
    ) -> QuestionGenerationResponse:
        """Generate clarifying questions for a persona concept.

        Args:
            request: The question generation request.

        Returns:
            QuestionGenerationResponse with questions.

        Raises:
            PromptGenerationException: If generation fails.
        """
        try:
            backend = self.llm_service.summarizer_backend

            if isinstance(backend, (OpenAIService, ZAIService)):
                raw_response = await self._generate_questions_openai(request)
            else:
                raw_response = await self._generate_questions_gemini(request)

            # Parse JSON response
            questions = self._parse_questions_response(raw_response)

            return QuestionGenerationResponse(
                success=True,
                questions=questions,
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Question generation failed: {e}", exc_info=True)
            return QuestionGenerationResponse(
                success=False,
                questions=[],
                error=str(e),
            )

    async def validate_prompt(self, prompt: str) -> tuple[bool, Optional[str]]:
        """Validate a system prompt.

        Args:
            prompt: The system prompt to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not prompt or not prompt.strip():
            return False, "프롬프트가 비어있습니다."

        # Check for minimum length
        if len(prompt) < 100:
            return False, "프롬프트가 너무 짧습니다. (최소 100자)"

        # Check for required sections (basic validation)
        required_keywords = ["role", "instruction", "설명", "역할"]
        prompt_lower = prompt.lower()
        if not any(keyword in prompt_lower for keyword in required_keywords):
            return False, "프롬프트에 역할 설명이 포함되어야 합니다."

        return True, None

    async def optimize_prompt_for_cache(self, prompt: str, target_tokens: int = 32768) -> str:
        """Optimize a prompt for context caching.

        Args:
            prompt: The prompt to optimize.
            target_tokens: Target token count for caching.

        Returns:
            Optimized prompt.
        """
        # TODO: Future enhancement - Implement prompt optimization for context caching
        # See issue #XXX for implementation roadmap
        # Consider: Prompt compression, key extraction, context window optimization
        # Currently returns prompt as-is without optimization
        return prompt

    async def _generate_prompt_from_concept(
        self, request: PromptGenerationRequest
    ) -> PromptGenerationResponse:
        """Generate prompt from concept without Q&A.

        Args:
            request: The prompt generation request.

        Returns:
            PromptGenerationResponse with generated prompt.
        """
        backend = self.llm_service.summarizer_backend

        if isinstance(backend, (OpenAIService, ZAIService)):
            result = await self._generate_prompt_openai(request.concept)
        else:
            result = await self._generate_prompt_gemini(request.concept)

        # Extract text from response
        text = self._extract_response_text(result)

        return PromptGenerationResponse(
            success=True,
            system_prompt=text,
        )

    async def _generate_prompt_with_answers(
        self, request: PromptGenerationRequest
    ) -> PromptGenerationResponse:
        """Generate prompt with interview answers incorporated.

        Args:
            request: The prompt generation request with Q&A.

        Returns:
            PromptGenerationResponse with generated prompt.
        """
        enhanced_concept = (
            f"**Original Concept:** {request.concept}\n\n"
            f"**Additional Details from Interview:**\n{request.questions_and_answers}\n\n"
            f"Use these interview answers to create a more personalized and detailed persona."
        )

        backend = self.llm_service.summarizer_backend

        if isinstance(backend, (OpenAIService, ZAIService)):
            result = await self._generate_prompt_openai(enhanced_concept)
        else:
            result = await self._generate_prompt_gemini(enhanced_concept)

        text = self._extract_response_text(result)

        return PromptGenerationResponse(
            success=True,
            system_prompt=text,
        )

    async def _generate_prompt_openai(self, concept: str) -> Any:
        """Generate prompt using OpenAI-style API.

        Args:
            concept: The persona concept.

        Returns:
            Raw API response.
        """
        backend = self.llm_service.summarizer_backend

        return await backend.execute_with_retry(
            lambda: backend.client.chat.completions.create(
                model=backend._summary_model_name,
                messages=[
                    {"role": "system", "content": META_PROMPT},
                    {"role": "user", "content": concept},
                ],
                temperature=getattr(self.config, "temperature", 1.0),
                top_p=getattr(self.config, "top_p", 1.0),
            ),
            "프롬프트 생성",
            timeout=60.0,
        )

    async def _generate_prompt_gemini(self, concept: str) -> Any:
        """Generate prompt using Gemini API.

        Args:
            concept: The persona concept.

        Returns:
            Raw API response.
        """
        backend = self.llm_service.summarizer_backend
        meta_model = backend.create_assistant_model(META_PROMPT, use_cache=False)

        return await backend.execute_with_retry(
            lambda: meta_model.generate_content(concept),
            "프롬프트 생성",
            timeout=60.0,
        )

    async def _generate_questions_openai(self, request: QuestionGenerationRequest) -> str:
        """Generate questions using OpenAI-style API.

        Args:
            request: The question generation request.

        Returns:
            Raw response text.
        """
        backend = self.llm_service.summarizer_backend

        raw_response = await backend.execute_with_retry(
            lambda: backend.client.chat.completions.create(
                model=backend._summary_model_name,
                messages=[
                    {"role": "system", "content": QUESTION_GENERATION_PROMPT},
                    {"role": "user", "content": request.concept},
                ],
                temperature=0.7,
            ),
            "질문 생성",
            timeout=60.0,
        )

        # Extract content from response
        if hasattr(raw_response, "choices") and raw_response.choices:
            return raw_response.choices[0].message.content
        return raw_response

    async def _generate_questions_gemini(self, request: QuestionGenerationRequest) -> str:
        """Generate questions using Gemini API.

        Args:
            request: The question generation request.

        Returns:
            Raw response text.
        """
        backend = self.llm_service.summarizer_backend
        question_model = backend.create_assistant_model(QUESTION_GENERATION_PROMPT, use_cache=False)

        result = await backend.execute_with_retry(
            lambda: question_model.generate_content(request.concept),
            "질문 생성",
            timeout=60.0,
        )

        # For Gemini, extract text from response
        if hasattr(result, "text"):
            return result.text
        return result

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from various response types.

        Args:
            response: The API response.

        Returns:
            Extracted text content.
        """
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        if isinstance(response, str):
            return response
        return str(response)

    def _parse_questions_response(self, response: str) -> list[dict[str, str]]:
        """Parse JSON response for questions.

        Args:
            response: The JSON response string.

        Returns:
            List of question dictionaries.
        """
        try:
            # Try to extract JSON from markdown code blocks
            if "```" in response:
                # Extract content between ```json and ```
                start = response.find("```json")
                if start == -1:
                    start = response.find("```")
                if start != -1:
                    start = response.find("\n", start) + 1
                    end = response.find("```", start)
                    if end != -1:
                        response = response[start:end].strip()

            data = json.loads(response)

            if "questions" in data:
                return data["questions"]

            # If direct list, wrap in expected format
            if isinstance(data, list):
                return data

            return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            return []

        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
            return []
