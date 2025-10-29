# Per-User Memory Implementation - Summary

## Overview
This document summarizes the comprehensive per-user memory system implementation for SoyeBot, enabling persistent, user-based memory storage with database persistence, semantic search capabilities, and AI-powered auto-extraction.

## Implementation Phases

### Phase 1: Database Layer & Schema Setup ✓
**Files Created/Modified:**
- `requirements.txt` - Added dependencies:
  - SQLAlchemy 2.0.23 (ORM)
  - alembic 1.13.1 (migrations)
  - sentence-transformers 2.2.2 (embeddings)
  - numpy 1.24.3 (vector operations)

- `config.py` - Added memory configuration:
  - `memory_model_name`: Lightweight model for memory operations
  - `memory_retrieval_mode`: 'inject_all' or 'semantic_search'
  - `max_memories_to_inject`: Default 20 memories
  - `embedding_model_name`: 'all-MiniLM-L6-v2' (80MB, CPU-friendly)
  - `database_path`: SQLite database location
  - `memory_cache_size`: LRU cache size
  - `max_conversation_history`: Max messages to keep per user
  - `memory_compression_days`: Age threshold for compression
  - `enable_memory_system`: Toggle for entire system

- `database/models.py` - SQLAlchemy models:
  - `User`: Discord user info (user_id, username, config_json)
  - `Memory`: Stores memories with embeddings (type, content, importance)
  - `ConversationHistory`: Chat logs with role and timestamp
  - `InteractionPattern`: User stats (messages, sentiment, topics)

- `services/database_service.py` - Database operations:
  - User CRUD operations
  - Memory storage and retrieval
  - Conversation history management
  - Interaction pattern tracking
  - Cleanup utilities for old data

### Phase 2: Memory Service with Auto-Extraction ✓
**Files Created:**
- `services/memory_service.py` - Core memory management:
  - Function calling tools for Gemini
  - Memory encoding and embedding generation
  - Toggleable retrieval modes (inject_all vs semantic_search)
  - Lazy-loading of embedding models
  - Handlers for:
    - `save_user_fact()`: Save learned facts
    - `save_preference()`: Save user preferences
    - `save_key_memory()`: Save important moments
    - `update_interaction_pattern()`: Track interaction stats

### Phase 3: Retrieval System (Toggleable) ✓
**Implemented in `memory_service.py`:**
- **Inject-All Mode** (Default for 1GB RAM):
  - Loads recent N memories (configurable)
  - Simple, fast, minimal resource usage
  - Best for resource-constrained environments

- **Semantic Search Mode** (Optional):
  - Uses sentence-transformers embeddings
  - Lazy-loads model only when needed
  - Unloads model after use to free RAM
  - Cosine similarity search
  - Top-K retrieval (configurable)

- **Context Injection**:
  - `get_memories_context()`: Format memories for prompt injection
  - Organized by memory type for clarity

### Phase 4: Session Management Refactor ✓
**Files Modified:**
- `bot/session.py` - SessionManager refactoring:
  - Changed from message_id-based to user_id-based sessions
  - Automatic user registration in database
  - Memory context loading on session creation
  - Conversation history loading and restoration
  - Message persistence to database
  - System prompt building with user memories

### Phase 5: Bot Integration ✓
**Files Created/Modified:**
- `bot/cogs/assistant.py` - AssistantCog updates:
  - Updated to use new SessionManager API
  - Memory context injection
  - Function calling tool registration
  - Message saving to conversation history
  - Function call handling (detection and dispatch)

- `bot/cogs/memory.py` - New memory commands:
  - `!기억 [내용]` - Manually save memory
  - `!기억목록` - List user's memories
  - `!기억삭제 [id]` - Delete specific memory
  - `!기억초기화` - Clear all memories (with confirmation)
  - `!기억설정` - Toggle retrieval mode
  - `!기억통계` - Show memory statistics

- `services/gemini_service.py` - Function calling support:
  - Updated `generate_chat_response()` to support tools
  - Added `parse_function_calls()` for response parsing

- `prompts.py` - Memory context template:
  - `MEMORY_CONTEXT_TEMPLATE`: Format for injecting memories
  - `build_system_prompt_with_memory()`: Build prompts with context

### Phase 6: Resource Optimization ✓
**Files Created:**
- `services/optimization_service.py` - Resource management:
  - `MemoryCache`: LRU cache for frequent memories (max 50 entries)
  - `OptimizationService`: Cleanup task management
  - `MemoryCompressionService`: Old conversation compression (future)
  - `ResourceMonitor`: Memory/database usage monitoring
  - Background cleanup loop for database maintenance
  - Cache statistics and hit rate tracking

**Key Optimizations for 1GB RAM/1-Core vCPU:**
- Default to inject_all mode (no embedding model loaded)
- Lazy-load semantic search only when enabled
- LRU cache to minimize database queries
- Connection pooling (max 2 connections)
- Aggressive conversation history cleanup (7-day retention)
- Batch operations for efficiency
- Context manager for safe resource management

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Discord Bot (@mentions)               │
│                                                          │
│  AssistantCog (bot/cogs/assistant.py)                  │
│  - Handles @mentions                                   │
│  - Injects memory context                              │
│  - Registers function calling tools                    │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┬────────────┐
        │                            │            │
    ┌───▼──────────┐    ┌──────────▼─┐    ┌────▼──────────┐
    │ SessionMgr   │    │ Memory Cmd │    │ GeminiService │
    │ (user-based) │    │ (Memory)   │    │ (with tools)  │
    └───┬──────────┘    └──────┬─────┘    └────┬──────────┘
        │                      │              │
        └──────┬───────────────┴──────────────┘
               │
    ┌──────────▼──────────────────────────────┐
    │    MemoryService                        │
    │  - Auto-extraction with function calls  │
    │  - Dual retrieval modes                 │
    │  - Embedding management                 │
    └────────────┬─────────────────────────────┘
                 │
    ┌────────────┼─────────────────────────────┐
    │            │                             │
┌───▼──┐   ┌─────▼──────┐   ┌────────────┐   │
│ DB   │   │ Embedding  │   │ OptimizeSvc│   │
│Svc   │   │   Model    │   │ - Cache    │   │
└──────┘   │ (lazy-load)│   │ - Cleanup  │   │
           └────────────┘   └────────────┘   │
                                             │
                      ┌──────────────────────┘
                      │
           ┌──────────▼─────────────┐
           │  SQLite Database       │
           │  (soyebot.db)          │
           │                        │
           │  Tables:               │
           │  - users               │
           │  - memories            │
           │  - conversation_history│
           │  - interaction_patterns│
           └────────────────────────┘
```

## Data Flow Examples

### Example 1: User Saves a Memory (Hybrid Mode)
```
User: !기억 나는 프로그래밍을 좋아한다
  ↓
MemoryCog.save_memory()
  ↓
MemoryService.save_memory()
  ↓
DatabaseService.save_memory()
  ↓
SQLite: INSERT INTO memories (user_id, memory_type, content, ...)
```

### Example 2: User Mentions Bot (Memory-Enriched Chat)
```
User: @SoyeBot 너는 뭐 하니?
  ↓
AssistantCog.on_message()
  ↓
SessionManager.get_or_create(user_id)
  ↓
- Load user memories from DB
- Load conversation history from DB
- Build system prompt with context
- Create Gemini chat with function calling tools
  ↓
GeminiService.generate_chat_response(tools=...)
  ↓
Gemini: Returns response + optional function calls
  ↓
- Auto-extract and save memories if functions called
- Save conversation to DB
- Send response to Discord
```

### Example 3: Semantic Search Mode (Optional)
```
Memory Retrieval (if mode='semantic_search')
  ↓
Load embedding model (all-MiniLM-L6-v2)
  ↓
Encode query/current message
  ↓
Fetch all user memories from DB
  ↓
Compute cosine similarity scores
  ↓
Return top-K most relevant memories
  ↓
Unload embedding model (free RAM)
  ↓
Inject into system prompt
```

## Configuration Examples

### Default Configuration (1GB RAM optimized)
```python
memory_model_name: 'gemini-1.5-flash-8b'
memory_retrieval_mode: 'inject_all'  # Simple mode
max_memories_to_inject: 20
embedding_model_name: 'all-MiniLM-L6-v2'
database_path: 'soyebot.db'
memory_cache_size: 50
max_conversation_history: 50
memory_compression_days: 7
enable_memory_system: True
```

### Enable Semantic Search
```python
# In config or via command
memory_service.set_retrieval_mode('semantic_search')

# Now uses embeddings for intelligent memory retrieval
# Model is lazy-loaded and unloaded to save RAM
```

## API Reference

### MemoryService Methods
```python
# Save memory
save_memory(user_id, memory_type, content, importance_score)

# Retrieve memories
retrieve_memories(user_id, query=None, max_memories=20)

# Get context for prompt injection
get_memories_context(user_id, query=None) -> str

# Toggle retrieval mode
set_retrieval_mode(mode: str) -> bool

# Function call handlers
handle_save_user_fact(user_id, fact, category)
handle_save_preference(user_id, preference)
handle_save_key_memory(user_id, memory, importance)
handle_update_interaction_pattern(user_id, topic, sentiment)
```

### DatabaseService Methods
```python
# User operations
get_or_create_user(user_id, username)
get_user(user_id)
update_user_last_seen(user_id)

# Memory operations
save_memory(user_id, memory_type, content, importance_score, embedding)
get_memories(user_id, memory_type=None, limit=20)
delete_memory(memory_id)
delete_all_memories(user_id)

# Conversation history
save_conversation(user_id, session_id, role, content)
get_conversation_history(user_id, session_id=None, limit=50)
delete_old_conversations(days=7)

# Interaction patterns
get_or_create_interaction_pattern(user_id)
update_interaction_pattern(user_id, topic=None, sentiment=None)
get_interaction_pattern(user_id)

# Maintenance
cleanup_expired_data(conversation_retention_days=7)
get_database_stats()
```

### SessionManager Methods
```python
# Create or get session
get_or_create(user_id, username, message_id=None) -> (chat, user_id)

# Save message to history
save_message(user_id, session_id, role, content)

# Cleanup expired sessions
cleanup_expired()
```

## Memory Types

### Fact (`fact`)
- Learned information about the user
- Category: hobby, location, profession, education, interest, other
- Importance: 0.7 (high)
- Example: "User loves programming"

### Preference (`preference`)
- User preferences or settings
- How the bot should interact
- Importance: 0.8 (very high)
- Example: "Prefers concise responses"

### Key Memory (`key_memory`)
- Important moments or events
- User-provided importance level
- Importance: Configurable (1-10)
- Example: "My birthday is December 25"

### Interaction Pattern (`interaction_patterns` table)
- Aggregate statistics
- Topic frequencies
- Average sentiment
- Total message count

## Limitations & Future Improvements

### Current Limitations
1. Function call parsing is simplified (detects patterns in text)
2. Memory compression not yet implemented
3. No memory deduplication
4. Embeddings not used for clustering/analysis

### Future Improvements
1. Proper function call parsing from Gemini response objects
2. Memory summarization for old conversations
3. Duplicate memory detection and merging
4. Memory clustering for better organization
5. Export/backup functionality
6. Memory permissions (private/shared)
7. Integration with vector databases (Pinecone, Weaviate)

## Testing Checklist

- [ ] Database initialization (create tables)
- [ ] User registration and lookup
- [ ] Memory CRUD operations
- [ ] Conversation history saving/loading
- [ ] Embed model lazy loading/unloading
- [ ] Semantic search functionality
- [ ] Inject-all retrieval
- [ ] LRU cache efficiency
- [ ] Function calling detection
- [ ] Memory commands (!기억, !기억목록, etc.)
- [ ] Session management with memory context
- [ ] Cleanup task execution
- [ ] Resource monitoring
- [ ] Memory command permissions
- [ ] Edge cases (empty memories, large histories, etc.)

## Performance Notes

### Expected Performance (1GB RAM, 1-Core vCPU)
- Memory retrieval: < 100ms (inject_all mode)
- Semantic search: 500-1000ms (when enabled)
- Database operations: < 50ms
- Embedding model load: 2-3s (first use)
- Cache hit rate: 30-50% (after warmup)

### Resource Usage
- Base RAM: ~300-400MB
- Per user session: ~5-10MB
- Embedding model: ~150MB (when loaded)
- SQLite database: Grows ~1MB per 10,000 memories
- LRU cache: ~50 entries max

## Installation & Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Update main.py to initialize memory services:
   ```python
   from services.database_service import DatabaseService
   from services.memory_service import MemoryService

   db_service = DatabaseService(config.database_path)
   memory_service = MemoryService(
       db_service=db_service,
       retrieval_mode=config.memory_retrieval_mode,
       embedding_model_name=config.embedding_model_name,
   )
   ```

3. Pass services to cogs:
   ```python
   await bot.add_cog(AssistantCog(bot, config, gemini_service, session_manager, memory_service))
   await bot.add_cog(MemoryCog(bot, config, db_service, memory_service))
   ```

4. Start optimization service (optional):
   ```python
   optimization_service = OptimizationService(db_service)
   await optimization_service.start_cleanup_loop()
   ```

## Contributing

When modifying the memory system:
1. Update this summary document
2. Add unit tests for new features
3. Test with 1GB RAM constraint
4. Update database models with proper migrations
5. Document API changes
