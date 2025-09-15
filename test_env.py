from deep_translator import GoogleTranslator
from openai import OpenAI
import os

# --- Test 1: Deep Translator ---
try:
    translated = GoogleTranslator(source="en", target="hi").translate("Hello Farmer!")
    print("Deep Translator ✅:", translated)
except Exception as e:
    print("Deep Translator ❌:", e)

# --- Test 2: OpenAI ---
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI ❌: Missing OPENAI_API_KEY in environment")
    else:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello in Hindi"}],
            max_tokens=20,
        )
        print("OpenAI ✅:", resp.choices[0].message.content)
except Exception as e:
    print("OpenAI ❌:", e)
