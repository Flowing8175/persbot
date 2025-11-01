# Modal.com LLM Deployment Guide

This guide walks through deploying your fine-tuned Qwen3-4B LoRA model to Modal.com for OpenAI-compatible API inference.

## Prerequisites

✅ Modal.com account (free $30/month credit)
✅ Modal CLI installed
✅ Your LoRA adapter at `/data/data/com.termux/files/home/soyemodel/soyemodel/`
✅ Base model: `Qwen/Qwen3-4B-Instruct-2507`

---

## Step 1: Install Modal CLI

```bash
# Install Modal package
pip install modal

# Authenticate with Modal
modal token new
# Follow the prompts to create an API token
# This will save your credentials to ~/.modal
```

---

## Step 2: (Optional) Merge LoRA Locally First

If you want to merge your LoRA adapter with the base model before uploading (saves time in Modal):

```bash
cd /data/data/com.termux/files/home/pys/soyebot

# Install requirements if not already done
pip install peft transformers torch

# Run merger script
python scripts/merge_lora.py
# Output will be saved to /tmp/merged_model
```

This step is **optional** - Modal can load LoRA directly, but merging first is faster.

---

## Step 3: Prepare Modal Deployment

### Option A: Deploy LoRA Directly (Recommended - Faster Setup)

No additional preparation needed. The Modal script will download the base model on first run.

### Option B: Upload Merged Model to Modal

If you merged in Step 2, upload the merged model:

```bash
# Create a Modal volume for your model
modal volume create soyebot-models

# Upload merged model (takes ~5-10 minutes)
modal cp /tmp/merged_model modal_volume_handle://models/merged_model
```

---

## Step 4: Deploy to Modal

### Quick Deploy

```bash
cd /data/data/com.termux/files/home/pys/soyebot

# Deploy the inference server
modal deploy scripts/modal_inference_server.py
```

You'll see output like:
```
✓ Created app 'soyebot-llm'
✓ Deployed app 'soyebot-llm'

Web endpoint: https://username--soyebot-llm.modal.run
```

**Save this URL!** You'll need it for your bot's `.env`.

### Monitor Deployment

```bash
# View logs
modal logs -f username--soyebot-llm

# Check status
modal app list
```

---

## Step 5: Test the API

### Using cURL

```bash
curl -X POST https://username--soyebot-llm.modal.run/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "soyebot-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### Using Python

```python
import requests

api_url = "https://username--soyebot-llm.modal.run/v1/chat/completions"

response = requests.post(
    api_url,
    json={
        "model": "soyebot-model",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "temperature": 0.7,
        "max_tokens": 128,
    },
    timeout=60
)

print(response.json())
```

---

## Step 6: Update Oracle Bot .env

On your Oracle Cloud instance, update `.env`:

```bash
# .env
DISCORD_TOKEN=your_token_here

# Switch to external mode (using Modal)
LLM_MODE=external
LLM_ENDPOINT_URL=https://username--soyebot-llm.modal.run
```

---

## Step 7: Test Bot Integration

Run the bot and test with a mention:

```bash
python -m soyebot.main
```

In Discord:
```
@SoyeBot How are you doing?
```

---

## Monitoring & Costs

### View Usage

```bash
# Check current deployment status
modal app status username--soyebot-llm

# View metrics dashboard
open https://modal.com/apps/username--soyebot-llm
```

### Cost Estimation

- **Free tier**: $30/month credit
- **GPU rates**: ~$0.1-0.25 per hour (T4/A10)
- **Estimate**:
  - 1000 inferences/day = ~$1-2/month
  - Plenty of room in $30 free credit!

### Monitor Credits

1. Go to https://modal.com/account/billing
2. Check "Usage This Month"
3. Free tier: $30/month resets monthly

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Ensure `torch`, `transformers`, `peft` are in `modal_inference_server.py` imports.

### Issue: OOM (Out of Memory)

**Solution**:
- Switch to smaller GPU (T4 instead of A100)
- Load model in 4-bit quantization
- Use LoRA instead of merged model

### Issue: Slow inference

**Solution**:
- Make sure you're using the correct GPU (check Modal dashboard)
- Reduce `MAX_LENGTH` in `modal_inference_server.py`
- Pre-merge LoRA before uploading

### Issue: "Cannot connect to endpoint"

**Solution**:
- Verify Modal deployment succeeded: `modal app list`
- Check logs: `modal logs -f username--soyebot-llm`
- Ensure URL in `.env` is correct (including `https://`)

---

## Updating Your Model

If you train a new LoRA adapter:

```bash
# Re-deploy (Modal will rebuild with latest code)
modal deploy scripts/modal_inference_server.py

# Or just update the model file if using volumes:
modal cp /path/to/new/model modal_volume_handle://models/merged_model
```

---

## Next: Running the Bot

Once deployment is successful:

1. SSH into Oracle Cloud instance
2. Update `.env` with Modal URL
3. Run bot: `python -m soyebot.main`
4. Test in Discord!

---

## Resources

- [Modal Docs](https://modal.com/docs)
- [Modal GPU Pricing](https://modal.com/pricing)
- [Modal Community](https://modal.com/community)
- [Your Modal Dashboard](https://modal.com/apps)
