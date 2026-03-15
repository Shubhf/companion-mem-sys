"""
LLM Provider — Gemini integration for the companion memory system.

Provides a unified llm_fn that the ConversationManager uses for generation.
Uses the new google.genai SDK.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from google import genai
from google.genai import types

API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize client
client = genai.Client(api_key=API_KEY) if API_KEY else None

# Models to try in order (fallback chain)
MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def create_gemini_llm_fn(model_name: str = None, max_retries: int = 2):
    """
    Create an llm_fn compatible with ConversationManager.
    Tries models in fallback chain if quota is hit.
    """
    if not client:
        raise ValueError("GEMINI_API_KEY not set in environment or .env file")

    preferred_model = model_name or MODEL_CHAIN[0]

    def llm_fn(messages: list[dict]) -> str:
        system_prompt = ""
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg["content"])]
                ))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg["content"])]
                ))

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=0.7,
            max_output_tokens=512,
        )

        # Try models in chain
        models_to_try = [preferred_model] + [m for m in MODEL_CHAIN if m != preferred_model]

        last_error = None
        for model in models_to_try:
            for attempt in range(max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    if response.text:
                        return response.text.strip()
                    return "[Empty response from Gemini]"
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower():
                        # Rate limited — wait briefly then try next model
                        if attempt < max_retries:
                            time.sleep(2 * (attempt + 1))
                            continue
                        break  # Try next model
                    elif "404" in err_str or "not found" in err_str.lower():
                        break  # Model not available, try next
                    else:
                        raise  # Unexpected error

        raise RuntimeError(f"All models exhausted. Last error: {last_error}")

    return llm_fn


def test_connection():
    """Quick test to verify Gemini API works."""
    try:
        fn = create_gemini_llm_fn()
        result = fn([
            {"role": "system", "content": "Reply in exactly 5 words."},
            {"role": "user", "content": "Hi, how are you?"},
        ])
        print(f"Gemini connected! Response: {result}")
        return True
    except Exception as e:
        print(f"Gemini connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()
