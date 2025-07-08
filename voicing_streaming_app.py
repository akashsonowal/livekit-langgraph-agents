import asyncio
import logging
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from typing import Annotated, TypedDict
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.plugins import deepgram, elevenlabs, langchain, silero, noise_cancellation, openai

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger("voicing-agent")

load_dotenv()

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
        )
        self._fast_llm = filler_adapter
        self._fast_llm_prompt = llm.ChatMessage(
            role="system",
            content=[
                "Generate a short instant response to the user's message with 5 to 10 words.",
                "Do not answer the questions directly. Examples:, let me think about that, "
                "wait a moment, that's a good question, etc.",
            ],
        )
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage):
        # Create a short "silence filler" response to quickly acknowledge the user's input
        fast_llm_ctx = turn_ctx.copy(
            exclude_instructions=True, exclude_function_call=True
        ).truncate(max_items=3)
        fast_llm_ctx.items.insert(0, self._fast_llm_prompt)
        fast_llm_ctx.items.append(new_message)

        # # Intentionally not awaiting SpeechHandle to allow the main response generation to
        # # run concurrently
        # self.session.say(
        #     self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable(),
        #     add_to_chat_ctx=False,
        # )

        # Alternatively, if you want the reply to be aware of this "silence filler" response,
        # you can await the fast llm done and add the message to the turn context. But note
        # that not all llm supports completing from an existing assistant message.

        fast_llm_fut = asyncio.Future[str]()

        async def _fast_llm_reply() -> AsyncIterable[str]:
            filler_response: str = ""
            async for chunk in self._fast_llm.chat(chat_ctx=fast_llm_ctx).to_str_iterable():
                filler_response += chunk
                yield chunk
            fast_llm_fut.set_result(filler_response)

        self.session.say(_fast_llm_reply(), add_to_chat_ctx=False)

        filler_response = await fast_llm_fut
        logger.info(f"Fast response: {filler_response}")
        turn_ctx.add_message(role="assistant", content=filler_response, interrupted=False)
    
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
        vad=silero.VAD.load(),
    )
    await session.start(PreResponseAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))