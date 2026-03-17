"""
LLM-based fact extraction for complex Hinglish messages.

Handles multi-clause corrections, temporal updates, entity disambiguation,
and role changes that regex cannot parse.

Falls back to rule-based extraction if LLM is unavailable or rate-limited.
"""

import json
import re
import os

EXTRACTION_PROMPT = """You are a memory extraction engine for an AI companion.

Extract ALL atomic facts from this user message as a JSON array.
Each fact must have:
- "entity": who/what (use "user" for the user themselves, lowercase names for others)
- "attribute": the property (use: name, job, nickname, crush_name, girlfriend_name, partner, species, weight, routine, role, captain, friend, likes, city, etc.)
- "value": the current/new value
- "is_correction": true if this UPDATES or REPLACES an old fact
- "old_value": (optional) what the old value was, if mentioned
- "temporal": (optional) "current", "past", "historical" — when this fact applies

RULES:
1. Extract EVERY fact, even implicit ones.
2. If someone says "X pehle Y thi, ab Z hai" → extract TWO facts: one past, one current (correction).
3. If entities are disambiguated ("Spark hamster hai, rat alag hai uska naam Pixel") → extract separate entities.
4. Temporal weight changes → extract current weight with is_correction=true.
5. Routine changes → extract both old and new routines with temporal markers.
6. Role changes → update the role, keep the relationship.
7. Keep attribute names simple and consistent.

Message: "{message}"

Return ONLY a JSON array. No markdown, no explanation.
Examples:
- "Mera naam Arjun hai" → [{{"entity":"user","attribute":"name","value":"Arjun","is_correction":false}}]
- "Spark rat nahi, hamster hai" → [{{"entity":"spark","attribute":"species","value":"hamster","is_correction":true,"old_value":"rat"}}]
- "Weight pehle 110 tha ab 88 hai" → [{{"entity":"user","attribute":"weight","value":"88","is_correction":true,"old_value":"110","temporal":"current"}}]
"""


def create_gemini_extractor(gemini_client=None, api_key: str = None):
    """
    Create an LLM-based fact extractor using Gemini.
    Returns a function compatible with MemoryIngestionPipeline.llm_extract_fn.
    """
    try:
        from google import genai
        from google.genai import types

        if not gemini_client:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                return None
            gemini_client = genai.Client(api_key=key)

        def extract(message: str) -> list[dict]:
            """Extract facts from message using Gemini."""
            prompt = EXTRACTION_PROMPT.format(message=message)

            try:
                resp = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=1024,
                    ),
                )
                text = resp.text.strip() if resp.text else "[]"

                # Extract JSON array from response
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    facts = json.loads(match.group())
                    # Normalize
                    for f in facts:
                        f["entity"] = f.get("entity", "user").lower().strip()
                        f["attribute"] = f.get("attribute", "").lower().strip().replace(" ", "_")
                        f["value"] = f.get("value", "").strip()
                        f["is_correction"] = f.get("is_correction", False)
                    return [f for f in facts if f["entity"] and f["attribute"] and f["value"]]
                return []
            except Exception:
                return []  # Fallback to rule-based

        return extract

    except Exception:
        return None
