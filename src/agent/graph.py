from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
import uuid

saver = InMemorySaver()

async def chat_with_checkpoint(state: MessagesState, thread_id: str | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    checkpoint = await saver.load(thread_id)
    prior_messages = checkpoint.get("messages", []) if checkpoint else []
    all_messages = prior_messages + state["messages"]
    response = llm.invoke(all_messages)
    updated_messages = all_messages + [response]
    await saver.save(thread_id, {"messages": updated_messages})
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat_with_checkpoint)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=saver)
