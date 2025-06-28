# main.py
"""
Main application to run the voice AI agent.
Uses LiveKit Agents for audio I/O and LangGraph for conversation state.
"""

import asyncio
from dotenv import load_dotenv
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import silero  # For voice activity detection (VAD), optional

from config import Config
from transcription import get_transcriber
from voicing_app import get_llm_model
from tts import get_tts_engine

load_dotenv()  # Ensure environment is loaded

class Assistant(Agent):
    """A simple Agent with instructions (could be expanded with tools)."""
    def __init__(self):
        super().__init__(instructions="You are a helpful voice AI assistant.")

async def entrypoint(ctx: agents.JobContext):
    # Load configuration
    config = Config

    # Instantiate providers based on configuration
    stt_engine = get_transcriber(config)
    llm_engine = get_llm_model(config)
    tts_engine = get_tts_engine(config)

    # Wrap STT, LLM, TTS so they are compatible with LiveKit's pipeline
    # For demonstration, we use LiveKit's plugins for STT/TTS that can wrap calls to our engines.
    # (Alternatively, one could implement a custom plugin interface.)

    # from livekit.plugins import base
    class CustomSTT:
        async def transcribe(self, audio: bytes) -> str:
            # Call your transcriber (may be sync, so wrap in thread if needed)
            return stt_engine.transcribe(audio)

    class CustomLLM:
        async def generate(self, messages):
            # messages is a list of dicts with role and content
            last_user = messages[-1]['content']
            reply = llm_engine.get_response(last_user)
            return {"choices": [{"message": {"role": "assistant", "content": reply}}]}

    class CustomTTS:
        async def synthesize(self, text: str) -> bytes:
            return tts_engine.synthesize(text)

    # class CustomSTT(base.STT):  # subclass a base STT plugin
    #         async def transcribe(self, audio: bytes) -> str:
    #             # Delegate to our transcriber
    #             return stt_engine.transcribe(audio)
    # class CustomLLM(base.LLM):
    #     async def generate(self, messages):
    #         # messages is the list of messages including system/user/assistant
    #         last_user = messages[-1]['content']
    #         reply = llm_engine.get_response(last_user, thread_id=ctx.job_id)
    #         return {"choices": [{"message": {"role": "assistant", "content": reply}}]}
    # class CustomTTS(base.TTS):
    #     async def synthesize(self, text: str) -> bytes:
    #         return tts_engine.synthesize(text)

    # Create the LiveKit agent session
    session = AgentSession(
        stt=CustomSTT(),
        llm=CustomLLM(),
        tts=CustomTTS(),
        vad=silero.VAD.load()
    )

    # Start the session: this connects to the room and begins audio exchange
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=None  # configure noise cancellation if available
        )
    )
    # Initial greeting from the assistant
    await session.generate_reply(instructions="Greet the user and offer assistance.")

if __name__ == "__main__":
    # Run as a LiveKit agent worker
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))