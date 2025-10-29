"Gemini API service for SoyeBot."

import google.generativeai as genai
import asyncio
import time
import re
import logging
from typing import Optional, Tuple, Any

import discord

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT
from utils import DiscordUI

logger = logging.getLogger(__name__)

class GeminiService:
    """Gemini API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.summary_model = genai.GenerativeModel(
            config.model_name, system_instruction=SUMMARY_SYSTEM_INSTRUCTION
        )
        self.assistant_model = genai.GenerativeModel(
            config.model_name, system_instruction=BOT_PERSONA_PROMPT
        )
        logger.info(f"Gemini 모델 '{config.model_name}' 로드 완료.")

    def create_assistant_model(self, system_instruction: str) -> genai.GenerativeModel:
        """Create a new assistant model with custom system instruction.

        Args:
            system_instruction: Custom system instruction for the model

        Returns:
            GenerativeModel instance with the custom instruction
        """
        return genai.GenerativeModel(
            self.config.model_name,
            system_instruction=system_instruction
        )

    def _is_rate_limit_error(self, error_str: str) -> bool:
        return ("429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower())

    def _extract_retry_delay(self, error_str: str) -> Optional[float]:
        match = re.search(r'Please retry in ([0-9.]+)s', error_str, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'seconds:\s*(\d+)', error_str)
        if match:
            return float(match.group(1))
        return None

    async def _api_request_with_retry(
        self,
        model_call,
        error_prefix: str = "요청",
        return_full_response: bool = False
    ) -> Optional[Any]:
        """재시도 및 에러 처리를 포함한 API 요청 래퍼

        Args:
            model_call: The function to call
            error_prefix: Error message prefix
            return_full_response: If True, return full response object; if False, return text only

        Returns:
            Response text (str) or full response object depending on return_full_response flag
        """
        for attempt in range(1, self.config.api_max_retries + 1):
            try:
                # Call the model without threading - the Gemini API handles its own async
                result = model_call()

                # If result is a coroutine, await it
                if asyncio.iscoroutine(result):
                    response = await asyncio.wait_for(
                        result,
                        timeout=self.config.api_request_timeout,
                    )
                else:
                    # If it's a regular blocking call, run in thread
                    response = await asyncio.wait_for(
                        asyncio.to_thread(lambda: result),
                        timeout=self.config.api_request_timeout,
                    )

                if return_full_response:
                    return response
                else:
                    return response.text.strip()
            except asyncio.TimeoutError:
                logger.warning(f"Gemini API 타임아웃 ({attempt}/{self.config.api_max_retries})")
                if attempt < self.config.api_max_retries:
                    logger.info(f"API 타임아웃, 재시도 중...")
                    continue
                logger.error(f"❌ 에러: API 요청 시간 초과")
                return None
            except Exception as e:
                error_str = str(e)
                logger.error(f"Gemini API 에러 ({attempt}/{self.config.api_max_retries}): {e}", exc_info=True)

                if self._is_rate_limit_error(error_str):
                    delay = self._extract_retry_delay(error_str) or self.config.api_rate_limit_retry_after
                    logger.warning(f"API 쿼터 초과: {delay}초 후 재시도")
                    for remaining in range(int(delay), 0, -1):
                        logger.info(f"⏳ 소예봇 뇌 과부하! 조금만 기다려 주세요. ({remaining}초)")
                        await asyncio.sleep(1)
                    continue

                if attempt >= self.config.api_max_retries:
                    break
                logger.info(f"에러 발생, 재시도 중...")
                await asyncio.sleep(2)

        logger.error(f"❌ 에러: 최대 재시도 횟수({self.config.api_max_retries})를 초과했습니다.")
        return None

    async def summarize_text(self, text: str) -> Optional[str]:
        if not text.strip():
            return "요약할 메시지가 없습니다."
        logger.info("Gemini API로 요약 요청...")
        prompt = f"Discord 대화 내용:\n{text}"
        return await self._api_request_with_retry(
            lambda: self.summary_model.generate_content(prompt),
            "요약"
        )

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        tools: Optional[list] = None,
    ) -> Optional[Tuple[str, Optional[Any]]]:
        """Generate chat response with optional function calling support.

        Args:
            chat_session: Gemini chat session
            user_message: User message
            tools: Optional list of function calling tools

        Returns:
            Tuple of (response_text, response_object) or None
            response_object is the full response for parsing function calls when tools are provided
        """
        logger.info("Gemini API로 채팅 응답 요청...")

        def api_call():
            if tools:
                # Use function calling mode
                return chat_session.send_message(
                    user_message,
                    tools=tools,
                )
            else:
                # Regular mode
                return chat_session.send_message(user_message)

        response_obj = await self._api_request_with_retry(
            api_call,
            "응답 생성",
            return_full_response=True
        )

        if response_obj is None:
            return None

        # Extract text from response, handling function calls properly
        response_text = self._extract_text_from_response(response_obj)
        return (response_text, response_obj if tools else None)

    def _extract_text_from_response(self, response_obj) -> str:
        """Extract text from Gemini response, handling function calls properly.

        Args:
            response_obj: Gemini response object

        Returns:
            Extracted text from response, or empty string if only function calls
        """
        try:
            # Try direct text access first (for simple responses without function calls)
            if hasattr(response_obj, 'text'):
                try:
                    return response_obj.text.strip()
                except ValueError:
                    # This happens when response contains function_call parts
                    pass

            # Extract text from parts manually
            text_parts = []
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                for candidate in response_obj.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            # Only extract text parts, skip function_call parts
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)

            if text_parts:
                return ' '.join(text_parts).strip()

            # If no text parts found, return empty string
            # (this is normal when Gemini only returns function calls)
            return ""

        except Exception as e:
            logger.error(f"Failed to extract text from response: {e}", exc_info=True)
            return str(response_obj)

    def parse_function_calls(self, response_obj) -> list:
        """Parse function calls from Gemini response.

        Args:
            response_obj: Gemini response object

        Returns:
            List of function call dictionaries
        """
        function_calls = []
        try:
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                for candidate in response_obj.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                # Safely handle args which might be None
                                args = {}
                                if hasattr(part.function_call, 'args') and part.function_call.args:
                                    try:
                                        args = dict(part.function_call.args)
                                    except (TypeError, ValueError) as e:
                                        logger.debug(f"Failed to convert args to dict: {e}, using empty dict")
                                        args = {}

                                function_calls.append({
                                    'name': part.function_call.name if hasattr(part.function_call, 'name') else '',
                                    'args': args,
                                })
                                logger.debug(f"Parsed function call: {function_calls[-1]['name']} with args: {args}")
        except Exception as e:
            logger.error(f"Failed to parse function calls: {e}", exc_info=True)
        return function_calls
