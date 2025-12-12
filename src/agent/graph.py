from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
import uuid

saver = InMemorySaver()
llm = ChatGroq(model="llama-3.1-8b-instant")

def chat_with_checkpoint(state: MessagesState, *, config):
    try:
        # Graph automatically injects the entire conversation state
        all_messages = state["messages"]

        # Extract raw text for the LLM
        inputs = [m["content"] for m in all_messages]

        # Call LLM (sync model)
        response = llm.invoke(inputs)

        # Standard assistant message format
        message_obj = {
            "role": "assistant",
            "content": response
        }

        # Return ONLY new messages.
        # Graph + saver will append this and persist automatically.
        return {"messages": [message_obj]}

    except Exception as e:
        print("Error in chat node:", e)
        raise



builder = StateGraph(MessagesState)
builder.add_node("chat", chat_with_checkpoint)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=saver)
