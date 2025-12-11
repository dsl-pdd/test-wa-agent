from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Simple chat node
def chat(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
