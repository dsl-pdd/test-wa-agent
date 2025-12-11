from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Helper to convert LangChain messages to simple dicts
def serialize_message(msg: BaseMessage):
    return {
        "role": "ai" if isinstance(msg, AIMessage) else "user",
        "content": msg.content
    }

# Chat node
def chat(state: MessagesState):
    llm_response = llm.invoke(state["messages"])
    if isinstance(llm_response, list):
        messages = [serialize_message(m) for m in llm_response]
    else:
        messages = [serialize_message(llm_response)]
    return {"messages": messages}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
