from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

models_to_test = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-pro-latest",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

for model in models_to_test:
    try:
        print(f"Testing {model}...")
        response = client.models.generate_content(model=model, contents="Hello")
        print(f"Success with {model}!")
        break
    except Exception as e:
        print(f"Failed: {type(e).__name__} - {str(e)[:100]}")
