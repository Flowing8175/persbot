#!/usr/bin/env python3
"""
Test client for Modal LLM API endpoint.

Usage:
    python scripts/test_modal_endpoint.py https://your-workspace--soyebot-llm.modal.run
"""

import sys
import json
import time
import requests
from typing import Optional


def test_health(api_url: str) -> bool:
    """Test health endpoint."""
    try:
        print(f"🔍 Testing health endpoint: {api_url}/health")
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            print(f"✅ Health check passed: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_chat_completion(api_url: str, prompt: str = "Hello, how are you?") -> Optional[str]:
    """Test chat completion endpoint."""
    try:
        print(f"\n💬 Testing chat completion endpoint...")
        print(f"📝 Prompt: {prompt}")

        payload = {
            "model": "soyebot-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant named Soye."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 128,
            "stream": False,
        }

        start_time = time.time()
        response = requests.post(
            f"{api_url}/chat/completions",
            json=payload,
            timeout=120,  # Long timeout for inference
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()

            if "error" in result:
                print(f"❌ API error: {result['error']}")
                return None

            message = result.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            print(f"\n✅ Response received in {elapsed:.2f}s")
            print(f"📨 Assistant: {content}")

            usage = result.get("usage", {})
            print(f"\n📊 Token usage:")
            print(f"   Prompt: {usage.get('prompt_tokens', 'unknown')}")
            print(f"   Completion: {usage.get('completion_tokens', 'unknown')}")
            print(f"   Total: {usage.get('total_tokens', 'unknown')}")

            return content
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        print("❌ Request timeout (inference took too long)")
        return None
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return None


def run_interactive(api_url: str):
    """Interactive chat with the Model."""
    print(f"\n🎯 Starting interactive chat with Modal API")
    print(f"📍 API: {api_url}")
    print(f"💡 Type 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Goodbye! 👋")
                break

            if not user_input:
                continue

            response = test_chat_completion(api_url, user_input)
            if response:
                print(f"Assistant: {response}\n")
            else:
                print("Failed to get response. Try again.\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python test_modal_endpoint.py <modal_api_url> [--interactive]")
        print("\nExample:")
        print("  python test_modal_endpoint.py https://username--soyebot-llm.modal.run")
        print("  python test_modal_endpoint.py https://username--soyebot-llm.modal.run --interactive")
        sys.exit(1)

    api_url = sys.argv[1].rstrip("/")
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    print(f"🚀 Modal API Endpoint Test")
    print(f"📍 URL: {api_url}\n")

    # Test health
    if not test_health(api_url):
        print("\n⚠️  Health check failed. Is Modal deployment active?")
        sys.exit(1)

    # Test chat completion
    if not interactive:
        test_chat_completion(api_url)
    else:
        run_interactive(api_url)


if __name__ == "__main__":
    main()
