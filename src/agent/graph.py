from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage
import uuid

saver = InMemorySaver()
def serialize_message(msg: BaseMessage):
    return {
        "role": "ai" if isinstance(msg, AIMessage) else "user",
        "content": msg.content
    }
llm = ChatGroq(model="llama-3.1-8b-instant")

def chat_with_checkpoint(state: MessagesState, *, config):
    try:
        # LangGraph injects state automatically
        all_messages = state["messages"]  

        # Extract pure text (list of strings)
        inputs = [m["content"] for m in all_messages]

        # Call LLM
        response = llm.invoke(inputs)

        # Normalize into list
        if isinstance(response, list):
            serialized = [serialize_message(m) for m in response]
        else:
            serialized = [serialize_message(response)]

        # LangGraph expects:
        # return {"messages": [ {role, content}, ... ]}
        return {"messages": serialized}

    except Exception as e:
        print("Error in chat node:", e)
        raise



builder = StateGraph(MessagesState)
builder.add_node("chat", chat_with_checkpoint)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=saver)
