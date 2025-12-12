from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
from langgraph.checkpoint import InMemorySaver
from langgraph.checkpoint.base import CheckpointSaver
import uuid
from langchain_core.messages import BaseMessage, AIMessage

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

#for memory
saver: Checkpointer = InMemorySaver()

# Helper to convert LangChain messages to simple dicts
def serialize_message(msg: BaseMessage):
    return {
        "role": "ai" if isinstance(msg, AIMessage) else "user",
        "content": msg.content
    }

# Simple chat node
# def chat(state: MessagesState):
#     response = llm.invoke(state["messages"])
#     return {"messages": [response]}

#Chat node with checkpointer integration
async def chat_with_checkpoint(state: MessagesState, thread_id: str | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    checkpoint = await saver.load(thread_id)
    prior_messages = checkpoint.get("messages", []) if checkpoint else []
    all_messages = prior_messages + state["messages"]
    response = llm.invoke(all_messages)
    if isinstance(response, list):
        messages = [serialize_message(m) for m in response]
    else:
        messages = [serialize_message(response)]
    updated_messages = all_messages + messages
    await saver.save(thread_id, {"messages": updated_messages})
    return {"messages": messages}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
