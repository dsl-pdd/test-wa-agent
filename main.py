from langgraph.checkpoint import InMemorySaver
from langgraph.checkpoint.base import CheckpointSaver
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import graph, saver
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list
    thread_id: str | None = None

class ChatResponse(BaseModel):
    messages: list

saver: CheckpointSaver = InMemorySaver()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 2. Load existing state for this thread (or start fresh)
        thread_id = request.thread_id or str(uuid4())
        checkpoint = await saver.load(thread_id)  
        prior_messages = checkpoint.get("messages", []) if checkpoint else []
        # 3. Combine prior messages with the new ones from the request
        all_messages = prior_messages + request.messages
        # 4. Invoke the graph with the combined history
        result = graph.invoke({"messages": all_messages})
        # 5. Append agent's new messages to the conversation
        updated_messages = all_messages + result.get("messages", [])
        # 5. Persist the updated state (including any new messages the agent produced)
        await saver.save(thread_id, {"messages": updated_messages})
        return ChatResponse(messages=result.get("messages", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}