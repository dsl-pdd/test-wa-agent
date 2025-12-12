from langgraph.checkpoint.memory import InMemorySaver
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import graph
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list
    thread_id: str | None = None

class ChatResponse(BaseModel):
    messages: list

# Initialize the saver
saver = InMemorySaver()

@app.post("/chat")
async def chat(request: ChatRequest):
    print("---- Incoming request ----")
    print("Thread ID:", request.thread_id)
    print("Messages:", request.messages)
    print("--------------------------")
    try:
        # 1. Use provided thread_id or generate a new one
        thread_id = request.thread_id or str(uuid.uuid4())

        # 2. Load previous messages from the saver
        checkpoint = await saver.load(thread_id)
        prior_messages = checkpoint.get("messages", []) if checkpoint else []

        # 3. Combine prior messages with the new messages
        all_messages = prior_messages + request.messages

        # 4. Invoke the graph (async) with full conversation and thread_id
        result = await graph.ainvoke(
            {"messages": all_messages},
            config={"thread_id": thread_id}
        )

        # 5. Append agent messages and save updated state
        updated_messages = all_messages + result.get("messages", [])
        await saver.save(thread_id, {"messages": updated_messages})

        # 6. Return only the agent's new messages
        return ChatResponse(messages=result.get("messages", []))

    except Exception as e:
        print("Error in chat post:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
