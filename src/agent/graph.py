from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
<<<<<<< HEAD
from langgraph.checkpoint import InMemorySaver
from langgraph.checkpoint.base import CheckpointSaver
import uuid
=======
from langchain_core.messages import BaseMessage, AIMessage
>>>>>>> d87a95ebaca65c2082db2b2d3bfb28cb0a332298

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

<<<<<<< HEAD
#for memory
saver: Checkpointer = InMemorySaver()

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
    updated_messages = all_messages + [response]
    await saver.save(thread_id, {"messages": updated_messages})
    return {"messages": [response]}
=======
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
>>>>>>> d87a95ebaca65c2082db2b2d3bfb28cb0a332298

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()
