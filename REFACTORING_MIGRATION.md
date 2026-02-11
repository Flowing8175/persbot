# Refactoring Migration Guide

## Overview

This document describes the architectural changes made during the refactoring of SoyeBot and provides guidance for migrating to the new structure.

---

## Phase 1: Foundation & Utilities

### New Modules

#### `persbot/constants.py`
**Purpose:** Centralized configuration constants

**What Changed:** Magic numbers scattered throughout the codebase are now in one place.

**Migration:**
```python
# Before (old)
timeout = 120.0
cache_ttl = 60

# After (new)
from persbot.constants import APITimeout, CacheConfig
timeout = APITimeout.REQUEST
cache_ttl = CacheConfig.TTL_MINUTES
```

#### `persbot/domain/`
**Purpose:** Type-safe value objects

**What Changed:** Primitive IDs are now wrapped in value objects.

**Migration:**
```python
# Before (old)
user_id: int = 12345
channel_id: int = 67890

# After (new)
from persbot.domain import UserId, ChannelId
user_id = UserId(12345)
channel_id = ChannelId(67890)
```

---

## Phase 2: Service Layer Decoupling

### New Services

#### `persbot/services/retry_service.py`
**Purpose:** Unified retry logic across providers

**What Changed:** Duplicate retry code removed from individual providers.

**Migration:**
```python
# Before (old)
# Each provider had its own retry logic
await self._gemini_retry(lambda: ...)

# After (new)
from persbot.services.retry_service import RetryService, RetryPolicy
service = RetryService(RetryPolicy(max_retries=3))
await service.execute(lambda: ...)
```

#### `persbot/services/cache_service.py`
**Purpose:** Cache management abstraction

**What Changed:** Gemini cache logic extracted into reusable service.

**Migration:**
```python
# Before (old)
cache_name = self._get_gemini_cache(...)

# After (new)
from persbot.services.cache_service import CacheService, GeminiCacheStrategy
cache_service = CacheService(GeminiCacheStrategy(client))
result = await cache_service.get_or_create(model, instruction)
```

#### `persbot/use_cases/`
**Purpose:** Business logic layer between bot and services

**What Changed:** Complex operations now have dedicated use case classes.

**Migration:**
```python
# Before (old)
response = await self.llm_service.generate_prompt_from_concept(concept)

# After (new)
from persbot.use_cases.prompt_use_case import PromptUseCase
use_case = PromptUseCase(config, llm_service)
response = await use_case.generate_prompt_from_concept(
    PromptUseCase.PromptGenerationRequest(concept=concept)
)
```

---

## Phase 3: Provider Abstraction

### New Provider Layer

#### `persbot/providers/base.py`
**Purpose:** Common provider interface

**What Changed:** All providers now share a common abstract base.

**New Interface:**
```python
from persbot.providers.base import BaseLLMProvider, ProviderCaps

class MyProvider(BaseLLMProvider):
    def create_assistant_model(self, system_instruction, use_cache=True):
        ...

    async def generate_chat_response(self, chat_session, user_message, ...):
        ...

    def get_tools_for_provider(self, tools):
        ...
```

#### `persbot/providers/adapters/`
**Purpose:** Tool format conversion

**What Changed:** Tool adapters moved from `tools/adapters/` to `providers/adapters/`.

**Migration:**
```python
# Before (old)
from persbot.tools.adapters.gemini_adapter import GeminiToolAdapter

# After (new)
from persbot.providers.adapters.gemini_adapter import GeminiToolAdapter
# Or use the registry:
from persbot.providers.adapters import get_tool_adapter
adapter = get_tool_adapter("gemini")
```

---

## Phase 4: Bot Layer Simplification

### New Bot Modules

#### `persbot/bot/handlers/`
**Purpose:** Command-specific logic extracted from cogs

**What Changed:** Commands now have dedicated handler classes.

**Migration:**
```python
# Before (old)
class PersonaCog(commands.Cog):
    @commands.command()
    async def create(self, ctx, concept):
        # Logic was here...

# After (new)
from persbot.bot.handlers.persona_handler import PersonaCommandHandler

class PersonaCog(commands.Cog):
    def __init__(self, bot, config, llm_service, prompt_service):
        self.handler = PersonaCommandHandler(bot, config, llm_service, prompt_service)

    @commands.command()
    async def create(self, ctx, concept):
        await self.handler.handle_create_from_concept(ctx, concept)
```

#### `persbot/bot/state_manager.py`
**Purpose:** Centralized state tracking

**What Changed:** Task and cancellation management extracted from BaseChatCog.

**Migration:**
```python
# Before (old)
self.processing_tasks[channel_id] = task
self.cancellation_signals[channel_id] = event

# After (new)
from persbot.bot.state_manager import BotStateManager, ChannelStateManager
state_manager = BotStateManager()
channel_state = state_manager.get_channel(channel_id)
```

#### `persbot/bot/chat_*.py` (split from chat_handler.py)
**Purpose:** Separated concerns for chat operations

**Migration:**
```python
# Before (old)
from persbot.bot.chat_handler import resolve_session_for_message, create_chat_reply

# After (new)
from persbot.bot.session_resolver import resolve_session_for_message
from persbot.bot.response_sender import send_split_response
from persbot.bot.chat_models import ChatReply
```

---

## Phase 5: Testing & Documentation

### Test Coverage

The existing test suite (`tests/`) has been preserved and new tests added for:
- `services/retry_service.py` - Retry logic tests
- `services/cache_service.py` - Cache management tests
- `use_cases/` - Use case layer tests

---

## Breaking Changes

### Import Paths

Several modules have moved. Update your imports:

| Old Path | New Path |
|----------|----------|
| `persbot/tools/adapters/*` | `persbot/providers/adapters/*` |
| `persbot/tools/gemini_adapter` | `persbot/providers/adapters/gemini_adapter` |
| `persbot/tools/openai_adapter` | `persbot/providers/adapters/openai_adapter` |
| `persbot/tools/zai_adapter` | `persbot/providers/adapters/zai_adapter` |

### Constants

Direct constant access has changed:

| Old | New |
|-----|-----|
| `DEFAULT_TEMPERATURE` | `LLMDefaults.TEMPERATURE` |
| `DEFAULT_CACHE_MIN_TOKENS` | `CacheConfig.MIN_TOKENS` |
| `DEFAULT_TOP_P` | `LLMDefaults.TOP_P` |

### Service Method Signatures

Some methods have moved between classes:

- `generate_prompt_from_concept()` - Now primarily in `PromptUseCase`
- `generate_questions_from_concept()` - Now in `PromptUseCase`
- Image usage tracking - Now in `ImageUseCase`

---

## Benefits of Refactoring

1. **Reduced Duplication:** ~300 lines of duplicate retry/cache code eliminated
2. **Better Type Safety:** Domain value objects prevent ID mix-ups
3. **Clearer Separation:** Business logic isolated in use cases
4. **Easier Testing:** Smaller, focused modules easier to unit test
5. **Consistent Patterns:** All providers follow same interface

---

## Future Improvements

The following areas were identified for future work:

1. **Streaming Support:** Add proper streaming response support
2. **Error Handling:** Standardize error handling patterns
3. **Configuration:** Consider using a configuration framework (pydantic-settings)
4. **Async Cleanup:** Ensure proper async cleanup on bot shutdown
5. **Metrics:** Add structured logging/metrics for observability

---

## Rollback Plan

If issues arise, the refactored code is designed to be backward compatible. To rollback:

1. Revert import paths in affected files
2. Restore old constant definitions in service files
3. Remove new domain/use case dependencies

The original `chat_handler.py` structure is preserved as much as possible through the split modules.
