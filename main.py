from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import graph
from typing_extensions import Dict

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str
    thread_id: str

class ChatResponse(BaseModel):
    message: str

# === In-memory state store ===
state_store: Dict[str, dict] = {}

@app.post("/chat")
async def chat(request: ChatRequest):
    print("---- Incoming request ----")
    print("Thread ID:", request.thread_id)
    print("Messages:", request.messages)
    print("--------------------------")
    try:
        thread_id = request.thread_id
        user_input = request.messages

        # Initialize state if new
        if thread_id not in state_store:
            state_store[thread_id] = {"messages": []}

        # Append user message
        state_store[thread_id]["messages"].append(HumanMessage(content=user_input))

        # Invoke graph
        result = await graph.ainvoke(
            state_store[thread_id],
            config={"configurable": {"thread_id": thread_id}}
        )

        # 'result' is a dict: {"messages": [...]}
        ai_messages = result.get("messages", [])

        # Append AI messages to state store
        for m in ai_messages:
            if isinstance(m, dict):
                state_store[thread_id]["messages"].append(AIMessage(content=m["content"]))
            else:
                state_store[thread_id]["messages"].append(m)

        # Return last AI message
        if ai_messages:
            last_msg = ai_messages[-1]
            last_reply = last_msg["content"] if isinstance(last_msg, dict) else last_msg.content
        else:
            last_reply = "No response generated."

        return ChatResponse(message=last_reply)
    except Exception as e:
        print("Error in chat post:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
