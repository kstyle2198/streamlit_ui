from typing import List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatGroq(temperature=0, model_name="qwen-qwq-32b")  # qwen-qwq-32b, llama-3.3-70b-versatile



# -----------------------------
# Step 1: Define state and agent
# -----------------------------

class GraphState(MessagesState):
    context: List[str]
    generation: str = ""

def generate_agent(state):
    prompt = ChatPromptTemplate.from_messages([
        ("human", 
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {messages} 
        Context: {context} 
        Answer:"""),
    ])
    
    rag_chain = prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc for doc in docs)

    total_docs = format_docs(state['context'])
    generation = rag_chain.invoke({"context": total_docs, "messages": state['messages']})
    return {"generation": generation}


from typing import TypedDict
class AgentState(TypedDict):
    topic: str
    joke: str

async def generate_joke(state, config):
    topic = state["topic"]
    print("Writing joke...\n")
    joke_response = await llm.ainvoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}],
        config,
    )
    print()
    return {"joke": joke_response.content}

# -----------------------------
# Step 2: Build LangGraph
# -----------------------------

def rag_builder(state_type):
    rag = StateGraph(state_type)
    rag.add_node("generate_agent", generate_agent)
    rag.add_edge(START, "generate_agent")
    rag.add_edge("generate_agent", END)
    return rag.compile()

graph_app = rag_builder(GraphState)


def create_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_edge(START, "generate_joke")
    workflow.add_edge("generate_joke", END)
    return workflow.compile()

graph_async = create_graph()




# -----------------------------
# Step 3: FastAPI Input/Output Models
# -----------------------------

class InvokeRequest(BaseModel):
    messages: List[str]
    context: List[str]

class InvokeResponse(BaseModel):
    generation: str

import json
def sse_format(payload):
    return f"data: {json.dumps(payload)}\n\n"

# -----------------------------
# Step 4: FastAPI Endpoint
# -----------------------------

# Define FastAPI app
app = FastAPI()


@app.post("/invoke", response_model=InvokeResponse)
def invoke_graph(request: InvokeRequest):
    try:
        # Run the LangGraph with input state
        inputs = {
            "messages": request.messages,
            "context": request.context
        }
        result = graph_app.invoke(inputs)
        return InvokeResponse(generation=result['generation'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class ARequest(BaseModel):
    topic: str

from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


from fastapi.responses import StreamingResponse
@app.post("/generate", response_model=ARequest)
async def generate_content(request: ARequest):
    topic = request.topic

    async def stream_generator():
        thinking_started = False

        async for msg, metadata in graph_async.astream({"topic": topic}, stream_mode="messages",):
            node = metadata["langgraph_node"]
            if node == "generate_joke":
                if msg.content:
                    if thinking_started:
                        print("\n</thinking>\n")
                        thinking_started = False
                    print(msg.content, end="", flush=True)
                    yield sse_format(
                        {"content": msg.content, "type": "joke", "thinking": False}
                    )
                if "reasoning_content" in msg.additional_kwargs:
                    if not thinking_started:
                        print("<thinking>")
                        thinking_started = True
                    print(
                        msg.additional_kwargs["reasoning_content"], end="", flush=True
                    )
                    yield sse_format(
                        {
                            "content": msg.additional_kwargs["reasoning_content"],
                            "type": "joke",
                            "thinking": True,
                        }
                    )
            else: pass

    return StreamingResponse(stream_generator(), media_type="text/event-stream")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("graph:app", host="0.0.0.0", port=8000, reload=True, workers=1)
