from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv(override=True)

# Define FastAPI app
app = FastAPI()

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
    
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    rag_chain = prompt | llm | StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc for doc in docs)

    total_docs = format_docs(state['context'])
    generation = rag_chain.invoke({"context": total_docs, "messages": state['messages']})
    return {"generation": generation}

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

# -----------------------------
# Step 3: FastAPI Input/Output Models
# -----------------------------

class InvokeRequest(BaseModel):
    messages: List[str]
    context: List[str]

class InvokeResponse(BaseModel):
    generation: str

# -----------------------------
# Step 4: FastAPI Endpoint
# -----------------------------

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
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
