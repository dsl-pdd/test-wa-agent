from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
import uuid

saver = InMemorySaver()

async def chat_with_checkpoint(state: MessagesState, thread_id: str | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    try:
        checkpoint = await saver.load(thread_id)
        prior_messages = checkpoint.get("messages", []) if checkpoint else []
        all_messages = prior_messages + state["messages"]
        
        # Ensure messages are strings for the LLM
        all_messages_text = [m["content"] for m in all_messages]
        response = llm.invoke(all_messages_text)
        
        message_obj = {"role": "assistant", "content": response}
        updated_messages = all_messages + [message_obj]
        await saver.save(thread_id, {"messages": updated_messages})
        return {"messages": [message_obj]}
    except Exception as e:
        print("Error in chat node:", e)
        raise HTTPException(status_code=500, detail=str(e))


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_with_checkpoint)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=saver)
