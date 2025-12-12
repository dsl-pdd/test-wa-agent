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
        thread_id = request.thread_id or f"whatsapp_{uuid.uuid4()}"
        
        # Graph invocation (async) with thread_id in config
        result = await graph.ainvoke(
            {"messages": request.messages},
            config={"thread_id": thread_id}
        )
        
        return ChatResponse(messages=result.get("messages", []))

    except Exception as e:
        print("Error in chat post:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
