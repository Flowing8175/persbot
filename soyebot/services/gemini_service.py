"Gemini API service for SoyeBot."

import google.generativeai as genai
import asyncio
import time
import re
import logging
from typing import Optional, Tuple, Any
from datetime import timedelta

import discord

from config import AppConfig
from prompts import SUMMARY_SYSTEM_INSTRUCTION, BOT_PERSONA_PROMPT

logger = logging.getLogger(__name__)

class GeminiService:
    """Gemini API와의 모든 상호작용을 관리합니다."""

    def __init__(self, config: AppConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)

        # Create shared models to avoid repeated instantiation (model pooling pattern)
        self._summary_model = genai.GenerativeModel(
            config.model_name, system_instruction=SUMMARY_SYSTEM_INSTRUCTION
        )

        # Base chat model - sessions will be created from this
        self._base_chat_model = None  # Lazy-initialized per system prompt

        # Prompt caching support (if available in SDK version)
        self._cached_prompt = None
        self._cache_supported = self._check_cache_support()

        if self._cache_supported:
            logger.info("Gemini prompt caching is supported - will attempt to use caching")
            self._initialize_prompt_cache()
        else:
            logger.info("Gemini prompt caching not available in current SDK version")

        logger.info(f"Gemini 모델 '{config.model_name}' 로드 완료.")

    def _check_cache_support(self) -> bool:
        """Check if the current SDK version supports prompt caching."""
        try:
            # Check if caching module exists
            from google.generativeai import caching
            return hasattr(caching, 'CachedContent')
        except (ImportError, AttributeError):
            return False

    def _initialize_prompt_cache(self):
        """Initialize prompt cache for the base persona.

        This can reduce token costs by ~75% and improve latency by 30-40%.
        """
        try:
            from google.generativeai import caching

            # Create cached content with 1-hour TTL
            self._cached_prompt = caching.CachedContent.create(
                model=self.config.model_name,
                system_instruction=BOT_PERSONA_PROMPT,
                ttl=timedelta(hours=1)
            )
            logger.info(f"Prompt cache created successfully (TTL: 1 hour)")
        except Exception as e:
            logger.warning(f"Failed to create prompt cache: {e}. Falling back to non-cached mode.")
            self._cache_supported = False
            self._cached_prompt = None

    def get_chat_session(self, system_instruction: str = None):
        """Get a new chat session from the shared base model.

        Uses prompt caching if available to reduce token costs by ~75%.
        Falls back to regular model instantiation if caching unavailable.

        Args:
            system_instruction: Custom system instruction for the model (optional)

        Returns:
            Chat session object
        """
        if system_instruction:
            # Create temporary model for custom system instruction
            model = genai.GenerativeModel(
                self.config.model_name,
                system_instruction=system_instruction
            )
            return model.start_chat()

        # Try to use cached prompt if available
        if self._cache_supported and self._cached_prompt:
            try:
                model = genai.GenerativeModel.from_cached_content(
                    cached_content=self._cached_prompt
                )
                logger.debug("Using cached prompt for chat session")
                return model.start_chat()
            except Exception as e:
                logger.warning(f"Failed to use cached prompt: {e}. Falling back to regular model.")
                # Fall through to regular model creation

        # Use shared base chat model for default persona (fallback)
        if self._base_chat_model is None:
            self._base_chat_model = genai.GenerativeModel(
                self.config.model_name,
                system_instruction=BOT_PERSONA_PROMPT
            )

        return self._base_chat_model.start_chat()

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

    def _log_raw_request(self, user_message: str, tools: Optional[list] = None, chat_session: Any = None) -> None:
        """Log raw API request data being sent.

        Args:
            user_message: The user message being sent
            tools: Optional function calling tools
            chat_session: The chat session object for logging history
        """
        # Skip expensive logging if debug level is not enabled
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(f"[RAW API REQUEST] User message length: {len(user_message)} characters")
            logger.debug(f"[RAW API REQUEST] User message content: {user_message}")
            logger.debug(f"[RAW API REQUEST] User message preview: {user_message[:200].replace(chr(10), ' ')}...")

            # Log tools if provided
            if tools:
                logger.debug(f"[RAW API REQUEST] Tools provided: {len(tools)} tool(s)")
                for tool_idx, tool in enumerate(tools):
                    tool_type = type(tool).__name__
                    logger.debug(f"[RAW API REQUEST] Tool {tool_idx}: {tool_type}")

                    # Log function declarations if available
                    if hasattr(tool, 'function_declarations'):
                        funcs = tool.function_declarations
                        logger.debug(f"[RAW API REQUEST]   Function declarations: {len(funcs)}")
                        for func_idx, func in enumerate(funcs):
                            func_name = func.name if hasattr(func, 'name') else 'unknown'
                            logger.debug(f"[RAW API REQUEST]   [{func_idx}] {func_name}")
                            if hasattr(func, 'description'):
                                logger.debug(f"[RAW API REQUEST]       Description: {func.description}")
                            if hasattr(func, 'parameters'):
                                params = func.parameters
                                if hasattr(params, 'properties'):
                                    logger.debug(f"[RAW API REQUEST]       Parameters: {list(params.properties.keys())}")
            else:
                logger.debug(f"[RAW API REQUEST] No tools provided - regular message mode")

            # Log chat session history if available
            if chat_session and hasattr(chat_session, 'history'):
                try:
                    history = chat_session.history
                    logger.debug(f"[RAW API REQUEST] Chat history: {len(history)} message(s)")
                    for msg_idx, msg in enumerate(history[-5:]):  # Log last 5 messages
                        role = msg.role if hasattr(msg, 'role') else 'unknown'
                        if hasattr(msg, 'parts') and msg.parts:
                            for part_idx, part in enumerate(msg.parts):
                                if hasattr(part, 'text') and part.text:
                                    text_preview = part.text[:100].replace('\n', ' ')
                                    logger.debug(f"[RAW API REQUEST]   [{msg_idx}] {role}: {text_preview}... ({len(part.text)} chars)")
                except Exception as e:
                    logger.debug(f"[RAW API REQUEST] Could not log chat history: {e}")

            logger.debug(f"[RAW API REQUEST] Raw request logging completed")

        except Exception as e:
            logger.error(f"[RAW API REQUEST] Error logging raw request: {e}", exc_info=True)

    def _log_raw_response(self, response_obj: Any, attempt: int) -> None:
        """Log raw API response data for debugging.

        Args:
            response_obj: The response object from Gemini API
            attempt: The current attempt number
        """
        # Skip expensive logging if debug level is not enabled
        if not logger.isEnabledFor(logging.DEBUG):
            return

        try:
            logger.debug(f"[RAW API RESPONSE {attempt}] Response type: {type(response_obj).__name__}")

            # Log candidates
            if hasattr(response_obj, 'candidates'):
                logger.debug(f"[RAW API RESPONSE {attempt}] Candidates count: {len(response_obj.candidates) if response_obj.candidates else 0}")

                if response_obj.candidates:
                    for candidate_idx, candidate in enumerate(response_obj.candidates):
                        logger.debug(f"[RAW API RESPONSE {attempt}] Candidate {candidate_idx}:")

                        # Log candidate finish reason
                        if hasattr(candidate, 'finish_reason'):
                            logger.debug(f"[RAW API RESPONSE {attempt}]   Finish reason: {candidate.finish_reason}")

                        # Log content parts
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            parts = candidate.content.parts
                            logger.debug(f"[RAW API RESPONSE {attempt}]   Content parts count: {len(parts)}")

                            for part_idx, part in enumerate(parts):
                                part_type = type(part).__name__
                                logger.debug(f"[RAW API RESPONSE {attempt}]   Part {part_idx}: {part_type}")

                                if hasattr(part, 'text'):
                                    text_preview = part.text[:100].replace('\n', ' ') if part.text else "(empty)"
                                    logger.debug(f"[RAW API RESPONSE {attempt}]     Text: {text_preview}... ({len(part.text)} chars)")

                                if hasattr(part, 'function_call') and part.function_call:
                                    func_name = part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'
                                    args_count = len(part.function_call.args) if hasattr(part.function_call, 'args') and part.function_call.args else 0
                                    logger.debug(f"[RAW API RESPONSE {attempt}]     Function call: {func_name}({args_count} args)")

                                    if hasattr(part.function_call, 'args') and part.function_call.args:
                                        args_dict = dict(part.function_call.args)
                                        logger.debug(f"[RAW API RESPONSE {attempt}]     Function args: {args_dict}")

            # Log usage data if available
            if hasattr(response_obj, 'usage_metadata'):
                metadata = response_obj.usage_metadata
                logger.debug(f"[RAW API RESPONSE {attempt}] Usage metadata:")
                if hasattr(metadata, 'prompt_token_count'):
                    logger.debug(f"[RAW API RESPONSE {attempt}]   Prompt tokens: {metadata.prompt_token_count}")
                if hasattr(metadata, 'candidates_token_count'):
                    logger.debug(f"[RAW API RESPONSE {attempt}]   Candidates tokens: {metadata.candidates_token_count}")
                if hasattr(metadata, 'total_token_count'):
                    logger.debug(f"[RAW API RESPONSE {attempt}]   Total tokens: {metadata.total_token_count}")

            logger.debug(f"[RAW API RESPONSE {attempt}] Raw response logging completed")

        except Exception as e:
            logger.error(f"[RAW API RESPONSE {attempt}] Error logging raw response: {e}", exc_info=True)

    async def _api_request_with_retry(
        self,
        model_call,
        error_prefix: str = "요청",
        return_full_response: bool = False,
        discord_message: Optional[discord.Message] = None,
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
                logger.debug(f"[API Request {attempt}/{self.config.api_max_retries}] Starting {error_prefix}...")

                # Call the model without threading - the Gemini API handles its own async
                result = model_call()

                # If result is a coroutine, await it
                if asyncio.iscoroutine(result):
                    logger.debug(f"[API Request {attempt}] Awaiting async result (timeout: {self.config.api_request_timeout}s)")
                    response = await asyncio.wait_for(
                        result,
                        timeout=self.config.api_request_timeout,
                    )
                else:
                    # If it's a regular blocking call, run in thread
                    logger.debug(f"[API Request {attempt}] Running blocking call in thread")
                    response = await asyncio.wait_for(
                        asyncio.to_thread(lambda: result),
                        timeout=self.config.api_request_timeout,
                    )

                logger.debug(f"[API Request {attempt}] Response received successfully")
                self._log_raw_response(response, attempt)

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
                if discord_message:
                    await discord_message.reply("❌ Gemini API 요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.", mention_author=False)
                return None
            except Exception as e:
                error_str = str(e)
                logger.error(f"Gemini API 에러 ({attempt}/{self.config.api_max_retries}): {e}", exc_info=True)

                if self._is_rate_limit_error(error_str):
                    delay = self._extract_retry_delay(error_str) or self.config.api_rate_limit_retry_after
                    sent_message = None
                    if discord_message:
                        sent_message = await discord_message.reply(
                            f"⏳ 소예봇 뇌 과부하! {int(delay)}초 후 재시도합니다.",
                            mention_author=False
                        )

                    # Update countdown every 5 seconds instead of every 1 second
                    # This reduces Discord API calls by 80% while maintaining UX
                    update_interval = 5
                    for remaining in range(int(delay), 0, -update_interval):
                        countdown_message = f"⏳ 소예봇 뇌 과부하! 조금만 기다려 주세요. ({remaining}초)"
                        if sent_message:
                            await sent_message.edit(content=countdown_message)
                        logger.info(countdown_message)
                        await asyncio.sleep(min(update_interval, remaining))

                    if sent_message:
                        await sent_message.delete()  # Delete the countdown message after it finishes
                    continue

                if attempt >= self.config.api_max_retries:
                    if discord_message:
                        await discord_message.reply(f"❌ API 요청 중 오류가 발생했습니다: {error_str}. 잠시 후 다시 시도해주세요.", mention_author=False)
                    break

                # Exponential backoff for transient errors
                backoff = min(
                    self.config.api_retry_backoff_base ** (attempt - 1),
                    self.config.api_retry_backoff_max
                )
                logger.info(f"에러 발생, {backoff}초 후 재시도 중... (attempt {attempt}/{self.config.api_max_retries})")
                await asyncio.sleep(backoff)

        logger.error(f"❌ 에러: 최대 재시도 횟수({self.config.api_max_retries})를 초과했습니다.")
        if discord_message:
            await discord_message.reply(f"❌ 최대 재시도 횟수({self.config.api_max_retries})를 초과하여 요청을 처리할 수 없습니다. 잠시 후 다시 시도해주세요.", mention_author=False)
        return None

    async def summarize_text(self, text: str) -> Optional[str]:
        if not text.strip():
            logger.debug("Summarization requested for empty text")
            return "요약할 메시지가 없습니다."
        logger.info(f"Summarizing text ({len(text)} characters)...")
        logger.debug(f"[RAW API REQUEST] Text to summarize:\n{text}")
        prompt = f"Discord 대화 내용:\n{text}"
        logger.debug(f"[RAW API REQUEST] Full prompt being sent ({len(prompt)} characters):\n{prompt}")
        return await self._api_request_with_retry(
            lambda: self._summary_model.generate_content(prompt),
            "요약"
        )

    async def generate_chat_response(
        self,
        chat_session,
        user_message: str,
        discord_message: discord.Message,
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
        logger.debug(f"Generating chat response - User message length: {len(user_message)}, Tools enabled: {tools is not None}")

        # Log raw request data
        self._log_raw_request(user_message, tools, chat_session)

        def api_call():
            if tools:
                # Use function calling mode
                logger.debug(f"[API REQUEST] Sending message with {len(tools)} tool(s)")
                return chat_session.send_message(
                    user_message,
                    tools=tools,
                )
            else:
                # Regular mode
                logger.debug(f"[API REQUEST] Sending message without tools")
                return chat_session.send_message(user_message)

        response_obj = await self._api_request_with_retry(
            api_call,
            "응답 생성",
            return_full_response=True,
            discord_message=discord_message,
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
            # Extract text from parts manually (safest approach for mixed responses)
            text_parts = []
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                logger.debug(f"Processing {len(response_obj.candidates)} candidates")
                for candidate in response_obj.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            # Only extract text parts, skip function_call parts
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                                logger.debug(f"Found text part: {len(part.text)} characters")
                            elif hasattr(part, 'function_call'):
                                logger.debug(f"Found function call part: {part.function_call.name if hasattr(part.function_call, 'name') else 'unknown'}")

            if text_parts:
                combined = ' '.join(text_parts).strip()
                logger.debug(f"Combined {len(text_parts)} text parts into {len(combined)} characters")
                return combined

            # If no text parts found, return empty string
            # (this is normal when Gemini only returns function calls)
            logger.debug("Response contains no text parts, only function calls")
            return ""

        except Exception as e:
            logger.error(f"Failed to extract text from response: {e}", exc_info=True)
            return ""
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
                logger.debug(f"Parsing function calls from {len(response_obj.candidates)} candidates")
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

                                func_name = part.function_call.name if hasattr(part.function_call, 'name') else ''
                                function_calls.append({
                                    'name': func_name,
                                    'args': args,
                                })
                                logger.debug(f"Parsed function call: {func_name} with {len(args)} args")
                logger.debug(f"Total function calls parsed: {len(function_calls)}")
        except Exception as e:
            logger.error(f"Failed to parse function calls: {e}", exc_info=True)
        return function_calls
