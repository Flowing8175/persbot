# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Persbot is a Discord AI bot that provides conversational AI capabilities using multiple LLM providers (Gemini, OpenAI, Z.AI). It features tool-based function calling, image generation, multi-model support, and session management. The bot is designed to run in resource-constrained environments (1GB RAM).

## Development Commands

### Running the Bot
```bash
python -m persbot.main
```

### Running in Background (Linux)
```bash
nohup python -m persbot.main > bot.log 2>&1 &
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_session.py

# Run with coverage
pytest --cov=persbot

# Run specific test function
pytest tests/test_buffer.py::test_buffer_delay
```

### Code Quality
```bash
# Format with ruff
ruff format .

# Lint with ruff
ruff check .

# Type checking with mypy
mypy persbot/
```

## Architecture

### Core Components Flow

```
Discord Message Event
        ↓
AssistantCog (via chat_handler.py & buffer.py)
        ↓
SessionManager.get_or_create() - Get/create chat session
        ↓
ToolManager.execute_tools() - Execute any function calls
        ↓
LLMService.generate_chat_response() - Generate response
        ↓
Provider Backend (GeminiService/OpenAIService/ZAIService)
        ↓
Send reply to Discord
```

### Key Architectural Patterns

1. **Multi-Provider LLM Abstraction**: `LLMService` acts as a factory that selects between Gemini, OpenAI, and Z.AI backends. Each backend implements `BaseLLMService` and provides provider-specific implementations for chat, tools, and function calling.

2. **Session Management**: `SessionManager` maintains channel-scoped chat sessions with automatic cleanup, model switching, and history tracking. Sessions are keyed as `channel:{channel_id}`.

3. **Tool System**: `ToolManager` coordinates three categories of tools:
   - **Discord Tools**: Read-only access to Discord data (channels, guilds, users)
   - **API Tools**: External APIs (weather, search, image generation, time)
   - **Persona Tools**: Immersion features (memory, media)

   Tools use adapters (`GeminiAdapter`, `OpenAIAdapter`, `ZaiAdapter`) to format function calls/results for each provider.

4. **Message Buffering**: `MessageBuffer` aggregates rapid messages within a delay window to prevent API spam, supporting cancellation via `asyncio.Event`.

5. **Model Usage Tracking**: `ModelUsageService` enforces daily usage limits per guild, with automatic fallback to alternate models when limits are exceeded (configured in `data/models.json`).

### Configuration System

Configuration is loaded from environment variables via `config.py`. The `.env.example` file documents all available options.

Key configuration areas:
- **LLM Provider Selection**: `ASSISTANT_LLM_PROVIDER`, `SUMMARIZER_LLM_PROVIDER`
- **Model Selection**: Provider-specific overrides (e.g., `GEMINI_ASSISTANT_MODEL_NAME`)
- **API Keys**: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ZAI_API_KEY`, `OPENROUTER_API_KEY`
- **Feature Flags**: `ENABLE_TOOLS`, `ENABLE_DISCORD_TOOLS`, `ENABLE_API_TOOLS`
- **Behavior Tuning**: `TEMPERATURE`, `TOP_P`, `MAX_HISTORY`, `MESSAGE_BUFFER_DELAY`

### Discord Cogs Structure

- `AssistantCog`: Main chat handler, processes @mentions and auto-reply channels
- `SummarizerCog`: Summarizes message history
- `ModelSelectorCog`: Per-channel model selection via `!model` command
- `PersonaCog`: Custom system prompts per channel
- `AutoChannelCog`: Auto-reply in configured channels

### Services Layer

- `services/llm_service.py`: Factory for provider backends
- `services/gemini_service.py`: Google Gemini implementation with context caching
- `services/openai_service.py`: OpenAI API implementation
- `services/zai_service.py`: Z.AI API implementation
- `services/prompt_service.py`: Prompt templates
- `services/model_usage_service.py`: Usage tracking and model definitions from `data/models.json`
- `services/image_model_service.py`: Image model selection per channel

### Important Data Files

- `data/models.json`: Defines available LLM models with daily limits, fallback chains, and image model preferences
- `.env`: Runtime configuration (not committed)

### Tool Execution Pattern

When the LLM requests a tool call:
1. Tool calls extracted via `extract_function_calls_from_response()`
2. `ToolManager.execute_tools()` runs tools in parallel via `asyncio.gather()`
3. Each tool is executed via `ToolExecutor.execute_tool()` with timeout and rate limit checks
4. Results formatted via `format_function_results_for_backend()`
5. Results sent back to LLM via `send_tool_results()`

### Testing Notes

- Tests mock external dependencies (`google.genai`, `openai`, `PIL`) at module level in `conftest.py`
- Uses `pytest-asyncio` with `auto` mode for async test handling
- `dpytest` provides Discord bot fixtures for integration testing
- Coverage target: comprehensive testing of core services (session, buffer, LLM, tools)
