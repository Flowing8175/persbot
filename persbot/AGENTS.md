# Persbot - Multi-Provider Discord AI Bot

> **AI Agent Guide**: This document provides hierarchical documentation for AI agents working on the Persbot codebase. Read the relevant sections based on your task scope.

## Quick Reference

| Component | Path | Purpose |
|-----------|------|---------|
| Entry Point | `main.py` | Bot initialization and startup |
| Configuration | `config.py` | Environment-based config loading |
| LLM Service | `services/llm_service.py` | Provider abstraction layer |
| Session Manager | `bot/session.py` | Chat session lifecycle |
| Tool Manager | `tools/manager.py` | Function calling orchestration |
| Assistant Cog | `bot/cogs/assistant/cog.py` | Main Discord bot commands |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Discord.py Bot                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │AssistantCog │  │ PersonaCog  │  │SummarizerCog│             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   SessionManager                           │  │
│  │  (LRU cache of ChatSession keyed by channel:user)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                        LLMService                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  ProviderRegistry                         │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │   │GeminiService│  │OpenAIService│  │  ZAIService │     │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          │                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ToolManager                            │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │   │DiscordTools │  │  APITools   │  │PersonaTools │     │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Bot Layer (`bot/`)

### Entry Point (`main.py`)

**Purpose**: Initialize Discord bot with required intents and register all cogs.

**Key Flow**:
1. Load config from environment (`load_config()`)
2. Create Discord bot with intents (messages, guilds, members, message_content)
3. Initialize services: `LLMService`, `SessionManager`, `PromptService`, `ToolManager`
4. Register cogs: `AssistantCog`, `PersonaCog`, `ModelSelectorCog`, `SummarizerCog`
5. Start bot with Discord token

**Important**: Member cache is optimized (`member_cache_flags`) to reduce memory footprint.

### Session Management (`bot/session.py`)

**Purpose**: Manage chat sessions with LRU eviction and periodic cleanup.

**Key Classes**:
- `ChatSession`: Holds chat object, user_id, session_id, model_alias
- `SessionContext`: Lightweight metadata for session reuse
- `SessionManager`: LRU cache with channel prompt overrides

**Session Key Format**: `channel:{channel_id}` (channel-wide sessions)

**Critical Methods**:
- `get_or_create()`: Returns cached session or creates new one with model compatibility check
- `set_session_model()`: Changes model preference, forces session reset on next interaction
- `undo_last_exchanges()`: Removes N user/assistant pairs from history

### Cogs (`bot/cogs/`)

#### AssistantCog (`assistant/cog.py`)

**Purpose**: Main AI assistant functionality via @mention.

**Commands**:
| Command | Aliases | Purpose |
|---------|---------|---------|
| `!retry` | 재생성, 다시 | Regenerate last response |
| `!stop` | 중단, 멈춰, abort | Cancel active processing |
| `!reset` | 초기화 | Clear channel session |
| `!temp` | - | Set temperature (0.0-2.0) |
| `!think` | 생각 | Set Gemini thinking budget |
| `!끊어치기` | - | Toggle break-cut streaming mode |
| `!delay` | 대기 | Set message buffer delay |

**Message Flow**:
1. `on_message()` → check if bot mentioned
2. Add to `message_buffer` for batching
3. `_process_batch()` → resolve session → generate response
4. `_send_response()` → streaming or chunked output

---

## Layer 2: Service Layer (`services/`)

### LLMService (`services/llm_service.py`)

**Purpose**: Factory and registry for LLM provider backends.

**Key Responsibilities**:
- Provider selection based on model alias
- Usage limit checking (`model_usage_service`)
- Tool result handling across providers
- Streaming response coordination

**Provider Resolution Flow**:
```python
model_alias → MODEL_DEFINITIONS[model_alias].provider → get_backend_for_model()
```

**Critical Methods**:
- `generate_chat_response()`: Main entry for non-streaming
- `generate_chat_response_stream()`: Main entry for streaming (yields chunks)
- `send_tool_results()`: Continue conversation after tool execution

### Provider Services

#### GeminiService (`services/gemini_service.py`)
- Uses `google-genai` SDK
- Supports context caching (`gemini_cache_min_tokens`)
- Supports thinking budget (`thinking_budget`)
- Streaming via async generator

#### OpenAIService (`services/openai_service.py`)
- Uses `openai` SDK
- Supports fine-tuned models (`openai_finetuned_model`)
- Service tier support (`service_tier`)
- Streaming via `stream=True` + async iterator

#### ZAIService (`services/zai_service.py`)
- Uses OpenAI-compatible API at `zai_base_url`
- Supports "coding plan" API endpoint
- Extended timeout (`zai_request_timeout: 300s`)

### PromptService (`services/prompt_service.py`)

**Purpose**: Load and cache system prompts from disk.

**Prompt Directory**: `persbot/assets/` (markdown files)

**Key Methods**:
- `get_prompt(name)`: Load prompt by filename (without .md)
- `get_summary_prompt()`: Get summarization system prompt

---

## Layer 3: Tools Layer (`tools/`)

### Tool Definition (`tools/base.py`)

**Purpose**: Define tool interface and data structures.

**Key Classes**:
- `ToolCategory`: Enum for tool organization (DISCORD_CHANNEL, API_SEARCH, etc.)
- `ToolParameter`: Parameter definition with type, required, default, enum
- `ToolResult`: Execution result with success, data, error, metadata
- `ToolDefinition`: Complete tool definition with `execute()` method

**Format Conversion**:
- `to_openai_format()`: Converts to OpenAI function calling schema
- `to_gemini_format()`: Converts to Gemini FunctionDeclaration

### ToolManager (`tools/manager.py`)

**Purpose**: Register, execute, and coordinate tools.

**Execution Flow**:
1. `execute_tools()` receives list of tool calls
2. Runs all tools in parallel via `asyncio.gather()`
3. Injects `discord_context` and API keys into parameters
4. Returns formatted results for provider

**Tool Categories**:
| Category | Path | Tools |
|----------|------|-------|
| Discord | `discord_tools/` | channel_info, user_info, guild_info |
| API | `api_tools/` | search, weather, time, image_generation |
| Persona | `persona_tools/` | memory, routine, media, web |

### Tool Adapters (`tools/adapters/`)

**Purpose**: Convert tool definitions to provider-specific formats.

- `gemini_adapter.py`: Converts to `genai_types.FunctionDeclaration`
- `openai_adapter.py`: Converts to OpenAI function schema format
- `zai_adapter.py`: Uses OpenAI-compatible format

---

## Layer 4: Domain Layer (`domain/`)

### Value Objects

**Provider** (`domain/model.py`):
```python
class Provider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ZAI = "zai"
```

**ModelAlias** (`domain/model.py`):
- Strongly-typed model identifier
- Auto-detects provider from name (e.g., "Gemini 2.5 Flash" → GEMINI)

**StandardModels**:
- Predefined model aliases: `GEMINI_FLASH`, `GPT_4O`, `GLM_4_7`, etc.

---

## Configuration Reference (`config.py`)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_TOKEN` | *required* | Discord bot token |
| `ASSISTANT_LLM_PROVIDER` | `gemini` | Provider for chat |
| `SUMMARIZER_LLM_PROVIDER` | `gemini` | Provider for summarization |
| `TEMPERATURE` | `1.0` | LLM temperature (0.0-2.0) |
| `TOP_P` | `1.0` | Nucleus sampling |
| `THINKING_BUDGET` | `off` | Gemini thinking tokens |
| `MAX_HISTORY` | `50` | Max conversation turns |
| `MESSAGE_BUFFER_DELAY` | `2.5` | Batch delay (seconds) |
| `ENABLE_TOOLS` | `true` | Enable function calling |
| `AUTO_REPLY_CHANNEL_IDS` | `` | Comma-separated channel IDs |

### Provider-Specific Models

Each provider has role-specific model configuration:
- `GEMINI_ASSISTANT_MODEL_NAME` / `GEMINI_SUMMARY_MODEL_NAME`
- `OPENAI_ASSISTANT_MODEL_NAME` / `OPENAI_SUMMARY_MODEL_NAME`
- `ZAI_ASSISTANT_MODEL_NAME` / `ZAI_SUMMARY_MODEL_NAME`

---

## Data Flow: Message Processing

```
1. Discord Message (on_message)
   │
   ├─► Should ignore? (bot message, no mention, etc.)
   │
   ├─► Add to MessageBuffer (debounce)
   │
   └─► _process_batch()
       │
       ├─► resolve_session() → session_key = "channel:{id}"
       │
       ├─► session_manager.get_or_create()
       │   └─► Create chat session with model_alias
       │
       ├─► llm_service.generate_chat_response_stream()
       │   │
       │   ├─► Check usage limits (model_usage_service)
       │   │
       │   ├─► Get backend for model alias
       │   │
       │   └─► backend.generate_chat_response_stream()
       │       │
       │       ├─► Send message with tools
       │       │
       │       ├─► Function calls? → execute_tools() → send_tool_results()
       │       │
       │       └─► Yield response chunks
       │
       └─► _send_response()
           ├─► Break-cut mode: split by newlines, send incrementally
           └─► Normal: accumulate then send
```

---

## Development Guidelines

### Adding a New Tool

1. Create tool handler in appropriate `tools/*/` directory:
```python
# tools/api_tools/my_tool.py
from persbot.tools.base import ToolCategory, ToolDefinition, ToolParameter, ToolResult

async def my_tool_handler(param1: str, discord_context=None) -> ToolResult:
    # Implementation
    return ToolResult(success=True, data={"result": "..."})

MY_TOOL = ToolDefinition(
    name="my_tool",
    description="Does something useful",
    category=ToolCategory.API_SEARCH,
    parameters=[
        ToolParameter(name="param1", type="string", description="...", required=True),
    ],
    handler=my_tool_handler,
)
```

2. Register in `tools/api_tools/__init__.py`:
```python
from .my_tool import MY_TOOL

def register_all_api_tools(registry):
    registry.register(MY_TOOL)
```

### Adding a New Provider

1. Create service in `services/my_provider_service.py`:
   - Extend `BaseLLMService` from `services/base.py`
   - Implement all abstract methods

2. Create model wrapper in `services/model_wrappers/my_provider_model.py`

3. Create session wrapper in `services/session_wrappers/my_provider_session.py`

4. Create tool adapter in `tools/adapters/my_provider_adapter.py`

5. Register in `services/llm_service.py`:
   - Add to `ProviderRegistry._create_provider()`
   - Add to `_is_provider_type()`

### Code Style

- **Type hints**: Required (enforced by mypy with `disallow_untyped_defs`)
- **Line length**: 100 characters
- **Formatting**: Ruff (quote-style: double, indent: space)
- **Imports**: Use absolute imports from `persbot.*`

### Error Handling

- Use `GENERIC_ERROR_MESSAGE` from `utils.py` for user-facing errors
- Log exceptions with `exc_info=True` for stack traces
- Return `ToolResult(success=False, error=...)` for tool failures

---

## Testing

### Running Tests
```bash
pytest
```

### Test Structure (`tests/`)
- Mirror source structure
- Use `pytest-asyncio` for async tests
- Use `dpytest` for Discord simulation
- Use `pytest-mock` for mocking

---

## Common Patterns

### Streaming Response Pattern
```python
async for chunk in llm_service.generate_chat_response_stream(
    chat_session, user_message, discord_message, tools=tools
):
    # Process chunk (send to Discord, accumulate, etc.)
    await send_chunk(chunk)
```

### Tool Execution Pattern
```python
tool_calls = [{"name": "search", "parameters": {"query": "..."}}]
results = await tool_manager.execute_tools(tool_calls, discord_context=message)
# results = [{"name": "search", "result": {...}}]
```

### Session Resolution Pattern
```python
resolved = await session_manager.resolve_session(
    channel_id=channel_id,
    author_id=author_id,
    username=username,
    message_id=message_id,
    message_content=content,
)
chat, session_key = await session_manager.get_or_create(
    user_id, username, resolved.session_key, channel_id, content
)
```

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `main.py` | Bot entry point, cog registration |
| `config.py` | Configuration loading, environment parsing |
| `prompts.py` | Default system prompts |
| `constants.py` | Enums, constants, meta-prompts |
| `exceptions.py` | Custom exception classes |
| `utils.py` | Helper functions, message sending |
| `rate_limiter.py` | Token bucket rate limiting |

---

## Version History

- **v0.1.0**: Initial release with multi-provider support
