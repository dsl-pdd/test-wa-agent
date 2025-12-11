from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import graph

from langchain_core.messages import BaseMessage

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list
    thread_id: str | None = None

class ChatResponse(BaseModel):
    messages: list

def serialize_message(msg: BaseMessage):
    return {
        "role": msg.type,
        "content": msg.content
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = graph.invoke({"messages": request.messages})

        # Convert LangGraph messages to dicts
        serialized = [serialize_message(m) for m in result["messages"]]

        return ChatResponse(messages=serialized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
