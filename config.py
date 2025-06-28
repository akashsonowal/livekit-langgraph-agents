# config.py
"""
Configuration module for the voice AI agent.
Loads API keys and provider selection from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file if present

class Config:
    # LiveKit server credentials (must be set for streaming)
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

    # Provider selection (e.g. 'openai', 'anthropic')
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    # Speech-to-Text provider (e.g. 'whisper', 'deepgram', 'assemblyai')
    STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper").lower()
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    # ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

    # Text-to-Speech provider (e.g. 'elevenlabs', 'google', 'amazon_polly')
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # path to service account JSON if using Google TTS
    # AMAZON_POLLY_REGION = os.getenv("AMAZON_POLLY_REGION", "us-east-1")
    # AMAZON_POLLY_ACCESS_KEY = os.getenv("AMAZON_POLLY_ACCESS_KEY")
    # AMAZON_POLLY_SECRET_KEY = os.getenv("AMAZON_POLLY_SECRET_KEY")