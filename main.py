from langchain_core.messages import HumanMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import graph

app = FastAPI()

class ChatRequest(BaseModel):
    messages: str
    thread_id: str

class ChatResponse(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    print("---- Incoming request ----")
    print("Thread ID:", request.thread_id)
    print("Messages:", request.messages)
    print("--------------------------")
    try:
        thread_id = request.thread_id
        user_input = request.messages

        # Create the input state with the new user message
        input_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        # Invoke graph with checkpointer - it will load previous state and save new state
        result = await graph.ainvoke(
            input_state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # Extract AI response
        ai_messages = result.get("messages", [])
        
        if ai_messages:
            last_msg = ai_messages[-1]
            last_reply = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        else:
            last_reply = "No response generated."

        return ChatResponse(message=last_reply)
    except Exception as e:
        print("Error in chat post:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
