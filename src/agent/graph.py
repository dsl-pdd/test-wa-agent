from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage

saver = InMemorySaver()

def serialize_message(msg: BaseMessage):
    return {
        "role": "ai" if isinstance(msg, AIMessage) else "user",
        "content": msg.content
    }

llm = ChatGroq(model="llama-3.1-8b-instant")

def chat_with_checkpoint(state: MessagesState, *, config):
    try:
        all_messages = state["messages"]
        thread_id = config.get("configurable", {}).get("thread_id")
        print("Chat node thread_id:", thread_id)
        print("Current message count:", len(all_messages))

        # Call LLM with all messages
        response = llm.invoke(all_messages)

        # Return the response - checkpointer will handle persistence automatically
        return {"messages": [response]}

    except Exception as e:
        print("Error in chat node:", e)
        raise

# Build StateGraph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_with_checkpoint)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=saver)
