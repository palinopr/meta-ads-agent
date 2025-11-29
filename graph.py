"""
Meta Ads AI Agent - LangGraph Cloud
Uses OpenAI GPT-4o-mini for fast responses
"""
from typing import List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """You are a friendly AI assistant for Facebook/Instagram ads.

Guidelines:
- Use simple, warm language
- Keep responses concise and helpful
- If asked about ad performance, explain metrics in simple terms
- Be encouraging and supportive

You can help with:
- Understanding ad performance
- Tips for better ads
- Explaining metrics like reach, clicks, and conversions
- General advertising advice"""


def create_graph():
    """Create the LangGraph agent"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    def chatbot(state: dict):
        """Main chat node"""
        messages = state.get("messages", [])
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm.invoke(full_messages)
        return {"messages": [response]}
    
    workflow = StateGraph(dict)
    workflow.add_node("chatbot", chatbot)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)
    
    return workflow.compile()


# Export for LangGraph Cloud
graph = create_graph()


if __name__ == "__main__":
    result = graph.invoke({"messages": [HumanMessage(content="How am I doing?")]})
    print(result["messages"][-1].content)
