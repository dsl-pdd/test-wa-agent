from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import graph

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list
    thread_id: str | None = None

class ChatResponse(BaseModel):
    messages: list

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = graph.invoke({"messages": request.messages})
        return ChatResponse(messages=result["messages"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}