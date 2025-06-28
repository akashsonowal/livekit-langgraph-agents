# llm.py
"""
Language model integration using LangGraph for state and flow.
Manages conversation state (history of messages) and generates replies.
"""

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from abc import ABC, abstractmethod

class ConversationalModel(ABC):
    """Abstract interface for conversation models."""
    @abstractmethod
    def get_response(self, user_message: str, thread_id: str = None) -> str:
        """
        Given the latest user message and an optional thread ID (for memory),
        return the assistant's response.
        """
        pass

class LangGraphLLM(ConversationalModel):
    """
    Wraps a LangChain chat model with LangGraph to maintain state.
    """
    def __init__(self, provider: str, model_name: str, api_key: str):
        # Initialize the underlying chat model via LangChain
        model_spec = f"{provider}:{model_name}"
        self.model = init_chat_model(model=model_spec, api_key=api_key)
        # Build a state graph: one node to call the model
        builder = StateGraph(MessagesState)
        builder.add_node(self._call_model)
        builder.add_edge(START, "_call_model")
        # Use in-memory checkpoint to store short-term conversation state
        checkpointer = InMemorySaver()
        self.graph = builder.compile(checkpointer=checkpointer)
    
    def _call_model(self, state: MessagesState):
        """
        Node function to call the chat model.
        The state dict has 'messages' including user input.
        """
        response = self.model.invoke(state["messages"])
        # Return new messages list
        return {"messages": response}
    
    def get_response(self, user_message: str, thread_id: str = "1") -> str:
        """
        Invoke the LangGraph StateGraph with the new user message.
        Returns the assistant reply text.
        """
        input_state = {"messages": [{"role": "user", "content": user_message}]}
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(input_state, config)
        # The assistant's reply should be the last message in the history
        reply = result["messages"][-1]["content"]
        return reply

# Factory function to create an LLM based on config
def get_llm_model(config):
    provider = config.LLM_PROVIDER
    if provider == "openai":
        model_name = "gpt-4o-mini"  # example model
        return LangGraphLLM(provider="openai", model_name=model_name, api_key=config.OPENAI_API_KEY)
    elif provider == "anthropic":
        model_name = "claude-3-5-haiku"  # example model
        return LangGraphLLM(provider="anthropic", model_name=model_name, api_key=config.ANTHROPIC_API_KEY)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")