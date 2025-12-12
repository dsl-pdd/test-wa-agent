from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, AIMessage

saver = InMemorySaver()

def serialize_message(msg: BaseMessage):
    return {
        "role": "ai" if isinstance(msg, AIMessage) else "user",
        "content": msg.content
    }

llm = ChatGroq(model="llama-3.1-8b-instant")

def chat_with_checkpoint(state: MessagesState, *, config):
    try:
        all_messages = state.messages  # messages list
        thread_id = getattr(state, "thread_id", None)
        print("Chat node thread_id:", thread_id)

        # Extract content for LLM
        inputs = [m["content"] for m in all_messages]

        # Call LLM
        response = llm.invoke(inputs)

        # Serialize response
        if isinstance(response, list):
            serialized = [serialize_message(m) for m in response]
        else:
            serialized = [serialize_message(response)]

        # Save messages in checkpoint per thread
        if thread_id:
            saver.save(thread_id, {"messages": all_messages + serialized})

        return {"messages": serialized}

    except Exception as e:
        print("Error in chat node:", e)
        raise

# Build StateGraph
builder = StateGraph(MessagesState)
builder.add_node("chat", chat_with_checkpoint)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile(checkpointer=saver)
