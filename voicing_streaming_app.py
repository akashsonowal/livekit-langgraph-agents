import os
import time
import asyncio
import logging
from datetime import datetime
from typing import Annotated, TypedDict
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    llm,
    MetricsCollectedEvent
)

from livekit.agents.metrics import (
    VADMetrics,
    EOUMetrics,
    STTMetrics,
    LLMMetrics,
    TTSMetrics,
    UsageCollector,
)

from livekit.plugins import deepgram, elevenlabs, langchain, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.llm.chat_context import ChatContext, ChatMessage

log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, "livekit_silent_filler_agent.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("livekit-langgraph-agent")

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# main and filler graph builders omitted for brevity (unchanged)
def build_main_graph():
    openai_llm = init_chat_model(model="gpt-4o-mini")
    def main_node(state: State):
        return {"messages": [openai_llm.invoke(state["messages"])]}
    builder = StateGraph(State)
    builder.add_node("main", main_node)
    builder.add_edge(START, "main")
    return builder.compile()

def build_filler_graph():
    fast_llm = init_chat_model(model="gpt-3.5-turbo")
    system_msg = ChatMessage(
        role="system",
        content=[
            "Generate a short (5-10 words) filler response like 'OK', 'Let me think'",
        ],
    )
    def filler_node(state: State):
        context = [system_msg] + state["messages"]
        response = fast_llm.invoke(context)
        return {"messages": state["messages"] + [response]}
    builder = StateGraph(State)
    builder.add_node("filler", filler_node)
    builder.add_edge(START, "filler")
    return builder.compile()

main_adapter = langchain.LLMAdapter(build_main_graph())
filler_adapter = langchain.LLMAdapter(build_filler_graph())

class PreResponseAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant",
            llm=main_adapter,
            tts=elevenlabs.TTS(),
        )
        self._fast_llm = filler_adapter
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=["Generate a very short instant response (5-10 words) without answering directly."],
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        fast_ctx = turn_ctx.copy(exclude_instructions=True, exclude_function_call=True).truncate(max_items=3)
        fast_ctx.items.insert(0, self._fast_llm_prompt)
        fast_ctx.items.append(new_message)

        fast_llm_fut = asyncio.Future()
        async def _fast_llm_reply() -> AsyncIterable[str]:
            start_time = time.time()
            response = ""
            async for chunk in self._fast_llm.chat(chat_ctx=fast_ctx).to_str_iterable():
                response += chunk
                yield chunk
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Fast LLM RTT: {elapsed:.2f} ms")
            fast_llm_fut.set_result(response)

        self.session.say(_fast_llm_reply(), add_to_chat_ctx=False)
        filler = await fast_llm_fut
        logger.info(f"Fast response: {filler}")
        turn_ctx.add_message(role="assistant", content=filler, interrupted=False)

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    current_turn_metrics = {
        'llm_ttft': None,
        'tts_ttfb': None,
        'stt_latency': None,
    }

    def calculate_total():
        if all(v is not None for v in current_turn_metrics.values()):
            llm_ms = current_turn_metrics['llm_ttft'] * 1000
            tts_ms = current_turn_metrics['tts_ttfb'] * 1000
            stt_ms = current_turn_metrics['stt_latency'] * 1000
            total = int(llm_ms + tts_ms + stt_ms)
            logger.info(
                "Total Conversation Latency",
                extra={
                    "total_ms": total,
                    "llm_ms": int(llm_ms),
                    "stt_ms": int(stt_ms),
                    "tts_ms": int(tts_ms),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            for k in current_turn_metrics:
                current_turn_metrics[k] = None

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3", language="multi"),
        tts=elevenlabs.TTS(),
        turn_detection=MultilingualModel(),
    )
    collector = UsageCollector()

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        m = ev.metrics
        if isinstance(m, LLMMetrics) and hasattr(m, 'ttft'):
            current_turn_metrics['llm_ttft'] = m.ttft
            calculate_total()
        elif isinstance(m, STTMetrics):
            # Streaming STT: ignore duration (always 0)
            if hasattr(m, 'audio_duration'):
                logger.debug(f"STT audio length: {m.audio_duration * 1000:.2f} ms")
        elif isinstance(m, EOUMetrics):
            if hasattr(m, 'transcription_delay'):
                current_turn_metrics['stt_latency'] = m.transcription_delay
                logger.info(
                    "STT Transcription Delay",
                    extra={"latency_ms": int(m.transcription_delay * 1000),
                           "timestamp": datetime.utcnow().isoformat()},
                )
                calculate_total()
        elif isinstance(m, TTSMetrics) and hasattr(m, 'ttfb'):
            current_turn_metrics['tts_ttfb'] = m.ttfb
            calculate_total()
        elif isinstance(m, VADMetrics):
            logger.debug(f"VAD Metrics: {m}")
        collector.collect(m)

    await session.start(
        agent=PreResponseAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    summary = collector.get_summary()
    logger.info(f"Usage summary: {summary}")

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
