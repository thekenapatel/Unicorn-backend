from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import traceback
from google import genai
from typing import List, Dict

load_dotenv()

app = FastAPI(title="UNICORN AI Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ CRITICAL: GEMINI_API_KEY is missing in .env file!")
else:
    print("✅ GEMINI_API_KEY loaded successfully")

client = genai.Client(api_key=api_key)


class ChatMessage(BaseModel):
    role: str
    parts: List[Dict[str, str]]


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


@app.get("/")
async def health():
    return {"status": "🦄 UNICORN AI Backend is running!"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"📨 Received request with {len(request.messages)} messages")

        contents = [{"role": msg.role, "parts": msg.parts} for msg in request.messages]

        # Most stable model combination as of April 2026
        models_to_try = [
            "gemini-3.1-flash-lite-preview",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-flash-lite-latest",
        ]

        for model_name in models_to_try:
            try:
                print(f"🔄 Trying model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                print(f"✅ Success with {model_name}")
                return {"response": response.text}
            except Exception as model_e:
                print(f"⚠️ {model_name} failed: {str(model_e)[:120]}")
                continue

        # If all models fail
        raise Exception("All Gemini models failed to respond")

    except Exception as e:
        print("🔴 GEMINI API ERROR:")
        traceback.print_exc()
        print(f"Error message: {str(e)}")

        user_msg = "Gemini is temporarily unavailable. Please wait 10-20 seconds and try again."
        raise HTTPException(status_code=500, detail=user_msg)
