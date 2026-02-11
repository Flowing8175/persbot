# Persbot - Multi-Provider Discord AI Bot

A feature-rich Discord bot powered by multiple LLM providers (Google Gemini, OpenAI, Z.AI) with chat assistance, summarization, persona management, and image generation capabilities.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Discord.py](https://img.shields.io/badge/discord.py-2.3.0+-blue.svg)](https://github.com/Rapptz/discord.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

### AI Chat Assistant
- **@mention responses**: Interact with the bot by mentioning @SoyeBot
- **Context-aware conversations**: Maintains session history for coherent multi-turn conversations
- **Customizable persona**: Configure bot personality via `prompts.py`

### Multi-LLM Provider Support
- **Google Gemini**: Default provider with flash and pro models
- **OpenAI**: Support for GPT models including fine-tuned models
- **Z.AI**: Support for GLM models with coding plan API option
- **Role-specific models**: Different models for assistant and summarization tasks

### Summarization
- Summarize Discord channel conversations
- Configurable message limits and time ranges
- Support for thread-specific summaries

### Image Generation
- AI-powered image generation via OpenRouter
- Multiple model options (Flux.2, Seedream, etc.)
- Rate limiting to control costs

### Tools System
- Discord integration tools (channel, guild, user info)
- API tools (search, weather, time, image generation)
- Persona tools (memory, routine, media handling)

### Development Mode
- Auto-reload on file changes during development
- Graceful shutdown handling

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/soyebot.git
cd soyebot
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

**Key dependencies:**
- `discord.py >= 2.3.0` - Discord API wrapper
- `google-genai >= 1.50.0` - Gemini API client
- `openai >= 1.59.6` - OpenAI API client
- `python-dotenv >= 1.0.0` - Environment configuration
- `sqlalchemy >= 2.0.44` - Database ORM

## Configuration

### 1. Environment variables

Copy `.env.example` to `.env` and configure:

```bash
# Required: Discord Bot Token
DISCORD_TOKEN=your_discord_bot_token_here

# LLM Provider Selection (gemini, openai, zai)
ASSISTANT_LLM_PROVIDER=gemini
SUMMARIZER_LLM_PROVIDER=gemini

# API Keys (configure based on your chosen provider)
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ZAI_API_KEY=your_zai_api_key_here

# Optional: Image Generation
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_IMAGE_MODEL=black-forest-labs/flux.2-klein-4b

# Optional: Auto-reply channels
AUTO_REPLY_CHANNEL_IDS=123456789,987654321
```

### 2. Discord Bot Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application and add a bot
3. Enable **Privileged Gateway Intents**:
   - Message Content Intent
   - Server Members Intent
   - Presence Intent
4. Generate an OAuth2 URL with `bot` scope and permissions:
   - Send Messages
   - Read Message History
   - Mention Everyone

### 3. API Key Setup

**Google Gemini:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Add to `.env` as `GEMINI_API_KEY`

**OpenAI:**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an API key
3. Add to `.env` as `OPENAI_API_KEY`

**Z.AI:**
1. Visit [Z.AI Platform](https://z.ai)
2. Create an API key
3. Add to `.env` as `ZAI_API_KEY`

## Usage

### Production Mode

```bash
python -m persbot.main
```

### Development Mode (Auto-Reload)

```bash
# Run with auto-restart on file changes
python run_dev.py

# Or with custom path/debounce settings
python run_dev.py --path persbot --debounce 0.5
```

### Background Service (Linux)

```bash
nohup python -m persbot.main > bot.log 2>&1 &
```

## Commands

### Prefix Commands
- `!summarize` - Summarize channel conversation
- `!summarize [minutes]` - Summarize last N minutes
- `!summarize [channel_id] threads` - Summarize specific channel

### Slash Commands
- `/model` - Select AI model/provider
- `/persona` - Manage bot persona
- `/help` - Show help message

## Project Structure

```
persbot/
├── main.py                    # Bot entry point
├── config.py                  # Configuration loader
├── prompts.py                 # AI system prompts
├── bot/
│   ├── cogs/
│   │   ├── assistant/         # AI chat assistant
│   │   ├── model_selector.py  # Model selection
│   │   ├── persona.py         # Persona management
│   │   ├── summarizer.py      # Summarization
│   │   └── auto_channel.py    # Auto-reply channels
│   ├── handlers/              # Event handlers
│   └── session.py             # Session management
├── services/
│   ├── llm_service.py         # LLM provider abstraction
│   ├── gemini_service.py      # Gemini integration
│   ├── openai_service.py      # OpenAI integration
│   ├── zai_service.py         # Z.AI integration
│   ├── prompt_service.py      # Prompt management
│   └── session_service.py     # Session persistence
├── tools/                     # Function calling tools
│   ├── discord_tools/         # Discord API tools
│   ├── api_tools/             # External API tools
│   └── persona_tools/         # Persona-specific tools
├── domain/                    # Domain models
├── use_cases/                 # Business logic
└── providers/                 # LLM provider adapters
```

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DISCORD_TOKEN` | *required* | Discord bot token |
| `ASSISTANT_LLM_PROVIDER` | `gemini` | LLM for chat (gemini/openai/zai) |
| `SUMMARIZER_LLM_PROVIDER` | `gemini` | LLM for summarization |
| `MAX_HISTORY` | `50` | Max conversation turns |
| `TEMPERATURE` | `1.0` | LLM temperature (0.0-2.0) |
| `MESSAGE_BUFFER_DELAY` | `2.5` | Message buffer delay (seconds) |
| `AUTO_REPLY_CHANNEL_IDS` | `` | Comma-separated channel IDs |

## System Requirements

- **Python**: 3.10 or higher
- **RAM**: 1GB minimum, 2GB recommended
- **CPU**: 1 vCPU minimum
- **Storage**: ~500MB for dependencies

## Troubleshooting

### Discord Intents Not Enabled
Make sure to enable **Message Content Intent** in the Discord Developer Portal.

### Rate Limiting
Adjust `MESSAGE_BUFFER_DELAY` in `.env` if experiencing rate limits.

### Session Issues
Sessions expire after 30 minutes of inactivity. Adjust `session_inactive_minutes` in `config.py`.

### API Errors
Check your API keys are valid and have sufficient quota.

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
ruff check persbot/
ruff format persbot/
```

---

Made with :heart: for high-performance Discord bots
