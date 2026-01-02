
from google.genai import types
import logging

# Configure logging to see the patch message
logging.basicConfig(level=logging.INFO)

# Import the service to trigger the patch
from soyebot.services.gemini_service import GeminiService

print("Testing Part.from_bytes with positional args...")
try:
    # This should now work if patched correctly
    # Positional args: data, mime_type
    p = types.Part.from_bytes(b"test data", "text/plain")
    print(f"Success: {p}")
except TypeError as e:
    print(f"Failed: {e}")
except Exception as e:
    print(f"Error: {e}")
