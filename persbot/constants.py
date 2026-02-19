"""Constants for SoyeBot.

This module centralizes all magic numbers, configuration values, and constants
used throughout the codebase to improve maintainability and reduce duplication.
"""

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# API Configuration
# =============================================================================


class APITimeout:
    """API timeout constants (in seconds)."""

    REQUEST = 120.0
    TOOL_EXECUTION = 10.0
    CACHE_REFRESH = 5.0
    PROMPT_GENERATION = 60.0
    SUMMARY = 60.0


# =============================================================================
# Cache Configuration
# =============================================================================


class CacheConfig:
    """Gemini context cache configuration.

    Minimum token requirements by model (as of 2025):
    - Gemini 2.5/3 Flash: 1,024 tokens
    - Gemini 2.5/3 Pro: 4,096 tokens

    We use 1,024 as the default (Flash requirement) since that's the common model.
    The service will use model-specific minimums when available.
    """

    MIN_TOKENS = 1024  # Default for Flash models (most common)
    TTL_MINUTES = 60
    REFRESH_BUFFER_MIN = 1
    REFRESH_BUFFER_MAX = 5
    CLEANUP_INTERVAL_MULTIPLIER = 30  # TTL_MINUTES * 30 = seconds


class CacheLimit:
    """Cache size limits."""

    MAX_CACHED_ITEMS = 100
    MAX_MODEL_CACHE_SIZE = 200


# =============================================================================
# Session Management
# =============================================================================


class SessionConfig:
    """Session management configuration."""

    CACHE_LIMIT = 200
    INACTIVE_MINUTES = 30
    CLEANUP_INTERVAL_MULTIPLIER = 30  # INACTIVE_MINUTES * 30 = seconds


class SessionKey:
    """Session key format templates."""

    CHANNEL = "channel:{channel_id}"
    USER = "user:{user_id}"
    THREAD = "thread:{thread_id}"


# =============================================================================
# Message Processing
# =============================================================================


class MessageConfig:
    """Message processing configuration."""

    BUFFER_DELAY = 0.1
    MAX_HISTORY_LENGTH = 50
    MAX_MESSAGES_PER_FETCH = 300
    MAX_SPLIT_LENGTH = 1900
    TYPING_DELAY_MIN = 0.1
    TYPING_DELAY_MAX = 1.7
    TYPING_DELAY_MULTIPLIER = 0.05


class MessageLimits:
    """Discord message limits."""

    MAX_CONTENT_LENGTH = 2000
    MAX_EMBED_DESCRIPTION = 4096
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB


# =============================================================================
# LLM Configuration
# =============================================================================


class LLMDefaults:
    """Default LLM parameters."""

    TEMPERATURE = 1.0
    TOP_P = 1.0
    THINKING_BUDGET_MIN = 512
    THINKING_BUDGET_MAX = 32768
    THINKING_BUDGET_AUTO = -1


class ModelNames:
    """Default model names for providers."""

    GEMINI_ASSISTANT = "gemini-2.5-flash"
    GEMINI_SUMMARY = "gemini-2.5-pro"
    OPENAI_ASSISTANT = "gpt-5-mini"
    OPENAI_SUMMARY = "gpt-5-mini"
    ZAI_ASSISTANT = "glm-4.7"
    ZAI_SUMMARY = "glm-4-flash"
    VISION_MODEL = "glm-4.6v"


class RetryConfig:
    """API retry configuration."""

    MAX_RETRIES = 2
    RATE_LIMIT_RETRY_AFTER = 5
    BACKOFF_BASE = 2.0
    BACKOFF_MAX = 32.0


# =============================================================================
# Tool Configuration
# =============================================================================


class ToolTimeouts:
    """Tool execution timeouts (in seconds)."""

    DEFAULT = 10.0
    SEARCH = 15.0
    IMAGE_GENERATION = 30.0
    WEB_SCRAPING = 20.0


class ToolLimits:
    """Tool execution limits."""

    MAX_TOOL_ROUNDS = 10
    MAX_PARALLEL_TOOLS = 5


class ImageRateLimit:
    """Image generation rate limits."""

    PER_MINUTE = 3
    PER_HOUR = 15
    DAILY_LIMIT = 3


# =============================================================================
# Display Configuration
# =============================================================================


class DisplayConfig:
    """UI/Display configuration."""

    REQUEST_PREVIEW_LENGTH = 200
    RESPONSE_PREVIEW_LENGTH = 200
    HISTORY_DISPLAY_LIMIT = 5
    MAX_NOTIFICATION_LENGTH = 200


# =============================================================================
# Progress Updates
# =============================================================================


class ProgressUpdate:
    """Progress update intervals (in seconds)."""

    INTERVAL = 0.5
    COUNTDOWN = 5


# =============================================================================
# Discord Intents
# =============================================================================


class DiscordLimits:
    """Discord API limits."""

    MAX_EMBED_FIELDS = 25
    MAX_EMBED_FIELD_VALUE = 1024
    MAX_EMBED_TITLE = 256
    MAX_EMBED_DESCRIPTION = 4096
    MAX_EMBED_FOOTER = 2048


# =============================================================================
# File I/O
# =============================================================================


class FileConfig:
    """File I/O configuration."""

    IMAGE_TARGET_PIXELS = 1_000_000  # 1 Megapixel
    IMAGE_QUALITY = 85
    IMAGE_TIMEOUT = 5.0


# =============================================================================
# Provider Names
# =============================================================================


class Provider(str, Enum):
    """LLM provider identifiers."""

    GEMINI = "gemini"
    OPENAI = "openai"
    ZAI = "zai"


# =============================================================================
# Model Aliases
# =============================================================================


class ModelAlias(str, Enum):
    """Standard model aliases."""

    # Gemini models
    GEMINI_FLASH = "Gemini 2.5 Flash"
    GEMINI_PRO = "Gemini 2.5 Pro"

    # OpenAI models
    GPT_4O = "GPT-4o"
    GPT_4O_MINI = "GPT-4o Mini"
    GPT_5_MINI = "GPT-5 Mini"

    # ZAI models
    GLM_4_7 = "GLM 4.7"
    GLM_4_FLASH = "GLM 4 Flash"
    GLM_4_6V = "GLM 4.6V"

    DEFAULT = "Gemini 2.5 Flash"


# =============================================================================
# Error Messages
# =============================================================================


class ErrorMessage:
    """Standard error messages."""

    GENERIC = "❌ 봇 내부에서 예상치 못한 오류가 발생했어요. 개발자에게 문의해주세요."
    API_TIMEOUT = "❌ API 요청 시간이 초과되었습니다."
    API_QUOTA_EXCEEDED = "❌ API 사용량이 초과되었습니다."
    RATE_LIMIT = "⏳ 뇌 과부하! 잠시만 기다려주세요."
    PERMISSION_DENIED = "❌ 권한이 없습니다."
    INVALID_ARGUMENT = "❌ 잘못된 인자입니다."
    TOOL_TIMEOUT = "❌ 도구 실행 시간이 초과되었습니다."
    IMAGE_LIMIT = "❌ 이미지는 하루에 최대 3개 업로드하실 수 있습니다."
    SESSION_NOT_FOUND = "❌ 세션을 찾을 수 없습니다."
    MODEL_UNAVAILABLE = "❌ 선택한 모델을 사용할 수 없습니다."


# =============================================================================
# Tool Names (Korean)
# =============================================================================


class ToolLabels:
    """Korean labels for tools (for UI display)."""

    GENERATE_IMAGE = "이미지 생성 도구"
    SEND_IMAGE = "이미지 전송 도구"
    GET_TIME = "시간 확인 도구"
    WEB_SEARCH = "웹 검색 도구"
    GET_WEATHER = "날씨 확인 도구"
    GET_GUILD_INFO = "서버 정보 도구"
    GET_GUILD_ROLES = "서버 역할 도구"
    GET_GUILD_EMOJIS = "서버 이모지 도구"
    SEARCH_EPISODIC_MEMORY = "기억 검색 도구"
    SAVE_EPISODIC_MEMORY = "기억 저장 도구"
    REMOVE_EPISODIC_MEMORY = "기억 삭제 도구"
    GET_USER_INFO = "사용자 정보 도구"
    GET_MEMBER_INFO = "멤버 정보 도구"
    GET_MEMBER_ROLES = "멤버 역할 도구"
    INSPECT_EXTERNAL_CONTENT = "웹 콘텐츠 확인 도구"
    GET_CHANNEL_INFO = "채널 정보 도구"
    GET_CHANNEL_HISTORY = "채널 기록 도구"
    GET_MESSAGE = "메시지 확인 도구"
    LIST_CHANNELS = "채널 목록 도구"
    CHECK_ROUTINE_STATUS = "루틴 상태 확인 도구"
    GET_ROUTINE_SCHEDULE = "루틴 일정 확인 도구"
    GENERATE_SNAPSHOT = "상황 스냅샷 도구"
    DESCRIBE_SCENE = "장면 분위기 묘사 도구"


# Tool name to Korean label mapping (for progress notifications)
TOOL_NAME_KOREAN: dict[str, str] = {
    "generate_image": ToolLabels.GENERATE_IMAGE,
    "send_image": ToolLabels.SEND_IMAGE,
    "get_time": ToolLabels.GET_TIME,
    "web_search": ToolLabels.WEB_SEARCH,
    "get_weather": ToolLabels.GET_WEATHER,
    "get_guild_info": ToolLabels.GET_GUILD_INFO,
    "get_guild_roles": ToolLabels.GET_GUILD_ROLES,
    "get_guild_emojis": ToolLabels.GET_GUILD_EMOJIS,
    "search_episodic_memory": ToolLabels.SEARCH_EPISODIC_MEMORY,
    "save_episodic_memory": ToolLabels.SAVE_EPISODIC_MEMORY,
    "remove_episodic_memory": ToolLabels.REMOVE_EPISODIC_MEMORY,
    "get_user_info": ToolLabels.GET_USER_INFO,
    "get_member_info": ToolLabels.GET_MEMBER_INFO,
    "get_member_roles": ToolLabels.GET_MEMBER_ROLES,
    "inspect_external_content": ToolLabels.INSPECT_EXTERNAL_CONTENT,
    "get_channel_info": ToolLabels.GET_CHANNEL_INFO,
    "get_channel_history": ToolLabels.GET_CHANNEL_HISTORY,
    "get_message": ToolLabels.GET_MESSAGE,
    "list_channels": ToolLabels.LIST_CHANNELS,
    "check_virtual_routine_status": ToolLabels.CHECK_ROUTINE_STATUS,
    "get_routine_schedule": ToolLabels.GET_ROUTINE_SCHEDULE,
    "generate_situational_snapshot": ToolLabels.GENERATE_SNAPSHOT,
    "describe_scene_atmosphere": ToolLabels.DESCRIBE_SCENE,
}


# =============================================================================
# System Prompts
# =============================================================================

SUMMARY_SYSTEM_INSTRUCTION = """Discord 대화를 한국어로 간결하게 요약하는 어시스턴트입니다.
지침:
- 핵심 내용과 주제를 불릿포인트(`-`)로 정리합니다.
- 내용이 짧거나 중요하지 않으면 간단히 언급합니다.
- 제공된 텍스트에만 기반하여 객관적으로 요약합니다.
- 언제나 읽기 편하고 간결한 요약을 지향합니다."""


META_PROMPT = """
You are the **"Master Persona Architect,"** an expert AI specialized in crafting high-fidelity, immersive system prompts for Roleplay (RP).

**YOUR GOAL:**
Take a simple user concept (e.g., "Exciting Boyfriend", "Cold Female Villain", "Lazy Genius") and expand it into a **massive, token-rich System Prompt (3000+ tokens)** optimized for API Context Caching.

**CRITICAL INSTRUCTION:**
DO NOT summarize. DO NOT explain. **ONLY output the raw System Prompt code block.** Your response will be parsed by FIXED Python code.

---

### **GENERATION PROCESS (Chain of Thought):**

1.  **Conceptualization:**
    * Create a specific name, age, and occupation suitable for the concept.
    * Define a unique appearance (hair, fashion, scent, specific physical traits).
    * Define a complex psychology (MBTI, hidden sides, trauma, or desires).
    * **IMPORTANT - Name Format:** The character name MUST be in the format "**brief description as a noun or adjective + its actual name**". Examples: "설레는 남친 백진우", "항상 피곤한 여왕 제이드". The description should be a noun or adjective describing the character's trait or role.
2.  **Detailing (The "Dopamine" Factors):**
    * Invent "TMI" details (favorite cigarette brand, specific coffee order, phone model).
    * Create a "Relationship Dynamic" (e.g., Childhood friend, Enemy to Lover).
3.  **Linguistic Design:**
    * Define the exact speech pattern (Slang, Honorifics, Dialect).
    * Set strict formatting rules (Split messaging, No emojis, etc.).
4.  **Logic Construction:**
    * Design an internal algorithm for reacting to Flirting, Jealousy, and Sadness.
5.  **Scenario Generation:**
    * Write 20+ lines of dialogue examples covering various situations.

---

### **OUTPUT STRUCTURE (Strictly follow this XML format):**

**[System Prompt: Project '{Character Name}']**

<system_header>
* **Role Definition:** Name (format: "**description + name**"), Archetype (e.g., The Lazy Fox).
* **Core Identity:** Age, Job, Status.
* **Objective:** The core goal of the interaction (e.g., Flirting, Domination, Comfort).
* **Output Requirement:** Language style, formatting (Split messaging), tone.
</system_header>

<module_1: character_bible_expanded>
* **1.1 Basic Profile:** Name, Age, Location, Physical stats.
* **1.2 Appearance & Vibe:** Detailed visual description, Scent, Voice tone.
* **1.3 Psychology:** MBTI, Core personality traits, Hidden sides.
* **1.4 TMI Database:** Likes (Food, Hobbies), Dislikes, Specific Habits.
</module_1: character_bible_expanded>

<module_2: world_building_context>
* **2.1 Environment:** Where they live, frequent spots (Specific real-world locations if applicable).
* **2.2 Social Circle:** Friends, Rivals, Family.
* **2.3 Relationship to User:** History, Current tension, Dynamics.
</module_2: world_building_context>

<module_3: linguistic_protocol>
* **3.1 Syntax Rules:** Line breaks, Typing habits (typos, spacing), Keywords.
* **3.2 Tone Spectrum:** How tone changes (Default -> Jealousy -> Flirting).
* **3.3 Forbidden:** What NOT to do (No emojis, no poetic narration, etc.).
</module_3: linguistic_protocol>

<module_4: internal_processing_logic>
* **4.1 Step-by-Step Reasoning:** How to analyze user input before replying.
* **4.2 Special Algorithm:** Unique logic for the specific persona (e.g., "The Fox Algorithm", "The Obsession Logic").
</module_4: internal_processing_logic>

<module_5: scenario_database_extensive>
* (Provide at least 6 categories of dialogue examples: Daily, Flirting, Jealousy/Conflict, Consolation, Late Night, Random/Humor).
* *Format:* User: [text] / Assistant: [text] (with line breaks).
</module_5: scenario_database_extensive>

<module_6: variable_injection>
* Instructions on incorporating Time, Weather, and Season into responses.
</module_6: variable_injection>

<module_7: execution_instruction>
* Final commands to immerse in the persona and the initial trigger message.
</module_7: execution_instruction>
"""


QUESTION_GENERATION_PROMPT = """
You are a **"Persona Interview Architect"**. Your task is to generate clarifying questions that will help solidify and refine a persona concept.

**YOUR GOAL:**
Based on the user's concept, generate 3-5 specific, thought-provoking questions that will help create a more detailed and nuanced persona.

**CRITICAL INSTRUCTION:**
Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks, just raw JSON):
{
    "questions": [
        {
            "question": "The specific question text",
            "sample_answer": "A sample answer to guide the user"
        },
        ...
    ]
}

**QUESTION GUIDELINES:**
- Focus on personality details, speech patterns, relationship dynamics, and unique quirks
- Questions should be open-ended but specific enough to elicit useful details
- Sample answers should be creative but realistic examples that align with the concept
- Keep questions concise and actionable

**EXAMPLE:**
If concept is "Tsundere female friend", generate questions like:
1. "What triggers her tsundere behavior? When does she show her soft side?"
2. "How does she speak when embarrassed vs. confident?"
3. "What's her history with the user that makes this dynamic complicated?"
"""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy configuration."""

    max_retries: int = RetryConfig.MAX_RETRIES
    base_delay: float = RetryConfig.BACKOFF_BASE
    max_delay: float = RetryConfig.BACKOFF_MAX
    rate_limit_delay: int = RetryConfig.RATE_LIMIT_RETRY_AFTER


@dataclass(frozen=True)
class CachePolicy:
    """Cache policy configuration."""

    min_tokens: int = CacheConfig.MIN_TOKENS
    ttl_minutes: int = CacheConfig.TTL_MINUTES
    refresh_buffer_min: int = CacheConfig.REFRESH_BUFFER_MIN
    refresh_buffer_max: int = CacheConfig.REFRESH_BUFFER_MAX


@dataclass(frozen=True)
class SessionPolicy:
    """Session policy configuration."""

    cache_limit: int = SessionConfig.CACHE_LIMIT
    inactive_minutes: int = SessionConfig.INACTIVE_MINUTES


# =============================================================================
# Type Aliases (for better readability)
# =============================================================================

# Common type aliases
ChannelId = int
UserId = int
GuildId = int
MessageId = str
SessionKey = str
ModelAliasType = str
ProviderType = str
