import asyncio
from typing import AsyncIterator, Optional, Dict, Any, List
from livekit.agents import llm
from livekit.agents.llm import ChatContext, ChatChunk, Choice, ChoiceDelta, CompletionUsage
from livekit.agents import Agent, AgentSession

# LangChain/LangGraph imports
from langchain_core.language_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage


class LiveKitLangGraphLLM(llm.LLM):
    """
    LiveKit LLM wrapper that uses LangChain chat model with LangGraph for state management.
    Integrates with LiveKit's streaming interface while maintaining conversation state.
    """
    
    def __init__(
        self, 
        provider: str, 
        model_name: str, 
        api_key: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ):
        super().__init__()
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the underlying chat model via LangChain
        model_spec = f"{provider}:{model_name}"
        self.model = init_chat_model(
            model=model_spec, 
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Build a state graph: one node to call the model
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", self._call_model)
        builder.add_edge(START, "call_model")
        
        # Use in-memory checkpoint to store short-term conversation state
        checkpointer = MemorySaver()
        self.graph = builder.compile(checkpointer=checkpointer)
        
        # Thread management for different conversations
        self._thread_counter = 0
        self._active_threads: Dict[str, str] = {}

    def _call_model(self, state: MessagesState):
        """
        Node function to call the chat model.
        The state dict has 'messages' including user input.
        """
        response = self.model.invoke(state["messages"])
        # Return new messages list - LangGraph will append to existing messages
        return {"messages": [response]}

    async def chat(
        self,
        *,
        chat_ctx: ChatContext,
        conn_options: llm.LLMConnOptions = llm.LLMConnOptions(),
        fnc_ctx: Optional[llm.FunctionContext] = None,
    ) -> "LangGraphLLMStream":
        """
        Main chat method that returns a stream of chat chunks.
        """
        return LangGraphLLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            fnc_ctx=fnc_ctx,
        )

    def _convert_livekit_to_langchain_messages(self, chat_ctx: ChatContext) -> List[Any]:
        """
        Convert LiveKit ChatContext to LangChain message format.
        """
        messages = []
        for msg in chat_ctx.messages:
            content = msg.content
            if isinstance(content, list):
                # Handle multimodal content - extract text for now
                text_content = ""
                for item in content:
                    if isinstance(item, str):
                        text_content += item
                content = text_content
            
            if msg.role == "user":
                messages.append(HumanMessage(content=str(content)))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=str(content)))
            # Add other role mappings as needed
        
        return messages

    def get_thread_id(self, chat_ctx: ChatContext) -> str:
        """
        Generate or retrieve thread ID for conversation state.
        You can customize this logic based on your needs.
        """
        # Simple approach: use hash of conversation context
        # In production, you might want to use user ID or session ID
        ctx_hash = str(hash(str(chat_ctx.messages)))
        if ctx_hash not in self._active_threads:
            self._thread_counter += 1
            self._active_threads[ctx_hash] = f"thread_{self._thread_counter}"
        return self._active_threads[ctx_hash]


class LangGraphLLMStream(llm.LLMStream):
    """
    Stream implementation for the LangGraph LLM.
    """
    
    def __init__(
        self,
        llm: LiveKitLangGraphLLM,
        chat_ctx: ChatContext,
        conn_options: llm.LLMConnOptions,
        fnc_ctx: Optional[llm.FunctionContext] = None,
    ):
        super().__init__(llm, chat_ctx, conn_options, fnc_ctx)
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._conn_options = conn_options
        self._fnc_ctx = fnc_ctx

    async def __aenter__(self) -> "LangGraphLLMStream":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self._stream_response()

    async def _stream_response(self) -> AsyncIterator[ChatChunk]:
        """
        Stream the response from LangGraph.
        Note: LangGraph doesn't natively support streaming, so we simulate it.
        """
        try:
            # Get the latest user message
            if not self._chat_ctx.messages:
                return
            
            latest_message = self._chat_ctx.messages[-1]
            if latest_message.role != "user":
                return
            
            # Convert to LangChain format
            langchain_messages = self._llm._convert_livekit_to_langchain_messages(self._chat_ctx)
            
            # Get thread ID for state management
            thread_id = self._llm.get_thread_id(self._chat_ctx)
            
            # Prepare input for LangGraph
            input_state = {"messages": langchain_messages}
            config = {"configurable": {"thread_id": thread_id}}
            
            # Invoke LangGraph (this maintains state)
            result = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._llm.graph.invoke(input_state, config)
            )
            
            # Get the assistant's reply
            if result and "messages" in result and result["messages"]:
                assistant_message = result["messages"][-1]
                response_text = assistant_message.content
                
                # Simulate streaming by yielding chunks
                # You can customize this to break text into smaller chunks
                chunk_size = 10  # words per chunk
                words = response_text.split()
                
                request_id = f"langgraph_{id(self)}"
                
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    # Add space at the end if not the last chunk
                    if i + chunk_size < len(words):
                        chunk_text += " "
                    
                    delta = ChoiceDelta(content=chunk_text)
                    choice = Choice(delta=delta, index=0)
                    
                    chunk = ChatChunk(
                        request_id=request_id,
                        choices=[choice],
                    )
                    
                    yield chunk
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.05)
                
                # Final chunk with usage info
                usage = CompletionUsage(
                    completion_tokens=len(words),
                    prompt_tokens=len(str(langchain_messages)),
                    total_tokens=len(words) + len(str(langchain_messages)),
                )
                
                final_chunk = ChatChunk(
                    request_id=request_id,
                    choices=[Choice(delta=ChoiceDelta(), index=0)],
                    usage=usage,
                )
                
                yield final_chunk
                
        except Exception as e:
            raise RuntimeError(f"LangGraph LLM error: {str(e)}")


# Alternative implementation for true streaming (if your LangChain model supports it)
class StreamingLangGraphLLM(LiveKitLangGraphLLM):
    """
    Enhanced version that supports true streaming if the underlying model does.
    """
    
    def __init__(self, provider: str, model_name: str, api_key: str, **kwargs):
        super().__init__(provider, model_name, api_key, **kwargs)
        
        # Try to enable streaming on the model if supported
        try:
            self.model = init_chat_model(
                model=f"{provider}:{model_name}",
                api_key=api_key,
                streaming=True,
                **kwargs
            )
        except Exception:
            # Fallback to non-streaming model
            pass

    async def _stream_langchain_response(self, messages: List[Any], thread_id: str) -> AsyncIterator[str]:
        """
        Stream response from LangChain model if streaming is supported.
        """
        try:
            # Check if model supports streaming
            if hasattr(self.model, 'astream'):
                async for chunk in self.model.astream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
            else:
                # Fallback to non-streaming
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.invoke(messages)
                )
                yield response.content
        except Exception as e:
            raise RuntimeError(f"Streaming error: {str(e)}")


# Usage examples
async def main():
    """
    Example usage of the LiveKit LangGraph LLM.
    """
    # Initialize the LLM
    llm = LiveKitLangGraphLLM(
        provider="openai",  # or "anthropic", "google", etc.
        model_name="gpt-4o-mini",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=1000,
    )
    
    # Test standalone usage
    print("Testing standalone usage...")
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hello! What's your name?")
    
    async with llm.chat(chat_ctx=chat_ctx) as stream:
        print("Assistant: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    
    # Test with conversation history (state management)
    print("Testing with conversation history...")
    chat_ctx.add_message(role="assistant", content="Hello! I'm Claude, an AI assistant.")
    chat_ctx.add_message(role="user", content="What did I just ask you?")
    
    async with llm.chat(chat_ctx=chat_ctx) as stream:
        print("Assistant: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


async def create_livekit_agent_with_langgraph():
    """
    Create a LiveKit agent using the LangGraph LLM.
    """
    # Initialize LangGraph LLM
    llm = LiveKitLangGraphLLM(
        provider="openai",
        model_name="gpt-4o-mini",
        api_key="your-openai-api-key",
    )
    
    # Add STT and TTS (you can use any providers)
    from livekit.plugins import openai as openai_plugin
    
    stt = openai_plugin.STT(model="whisper-1")
    tts = openai_plugin.TTS(model="tts-1", voice="alloy")
    
    # Create agent session
    session = AgentSession(
        llm=llm,
        stt=stt,
        tts=tts,
    )
    
    return session


# Example with custom state management
class CustomStateLangGraphLLM(LiveKitLangGraphLLM):
    """
    Extended version with custom state management logic.
    """
    
    def __init__(self, provider: str, model_name: str, api_key: str, **kwargs):
        super().__init__(provider, model_name, api_key, **kwargs)
        self.user_sessions: Dict[str, str] = {}
    
    def get_thread_id(self, chat_ctx: ChatContext, user_id: Optional[str] = None) -> str:
        """
        Custom thread ID generation based on user ID.
        """
        if user_id:
            if user_id not in self.user_sessions:
                self._thread_counter += 1
                self.user_sessions[user_id] = f"user_{user_id}_thread_{self._thread_counter}"
            return self.user_sessions[user_id]
        else:
            return super().get_thread_id(chat_ctx)
    
    def clear_user_session(self, user_id: str):
        """
        Clear conversation state for a specific user.
        """
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]


if __name__ == "__main__":
    asyncio.run(main())