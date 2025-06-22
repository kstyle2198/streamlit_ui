import re
from typing import List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import FlagReranker


from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0,max_tokens=3000,)  # qwen-qwq-32b, llama-3.3-70b-versatile
llm_think = ChatGroq(model_name="qwen-qwq-32b", temperature=0,max_tokens=3000,)  # qwen-qwq-32b, llama-3.3-70b-versatile

from FlagEmbedding import FlagReranker
reranking_model_path = "D:/LLMs/bge-reranker-v2-m3"
reranker = FlagReranker(model_name_or_path=reranking_model_path, 
                        use_fp16=True,
                        batch_size=512,
                        max_length=2048,
                        normalize=True)

from typing import Union, List
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy

def load_elastic_vectorstore(index_names: Union[str, List[str]]):
    # 단일 문자열인 경우 리스트로 변환
    if isinstance(index_names, str):
        index_names = [index_names]
    
    vector_store = ElasticsearchStore(
        index_name=index_names, 
        embedding=OllamaEmbeddings(
            base_url="http://localhost:11434", 
            model="bge-m3:latest"
        ), 
        es_url="http://localhost:9200",
        es_user="Kstyle",
        es_password="12345",
    )
    return vector_store


index_names = ["ship_safety"]
vector_store = load_elastic_vectorstore(index_names=index_names)

import heapq
from langchain_core.documents import Document

def reranking(query: str, docs: list, min_score: float = 0.5, top_k: int = 3):
    """
    doc string
    """
    global reranker
    inputs = [[query.lower(), doc['page_content'].lower()] for doc in docs]
    scores = reranker.compute_score(inputs)
    if not isinstance(scores, list):
        scores = [scores]

    print(f"---- original scores: {scores}")

    # Filter scores by threshold and keep index
    filtered_scores = [(score, idx) for idx, score in enumerate(scores) if score >= min_score]

    # Get top_k using heapq (more efficient than sorting full list)
    top_scores = heapq.nlargest(top_k, filtered_scores, key=lambda x: x[0])

    # Get document objects from top indices
    reranked_docs = [docs[idx] for _, idx in top_scores]

    return top_scores, reranked_docs

# -----------------------------
# Step 1: Define state and agent
# -----------------------------
import operator
from typing import Annotated
from typing import Dict, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
class OverAllState(TypedDict):
    """
    Represents the overall state of our graph.
    """
    question: str
    questions: Annotated[list, operator.add]
    context: Annotated[list, operator.add]
    rerank_context: Annotated[list, operator.add]
    top_scores: Annotated[list, operator.add]
    top_k: Dict = {'doc':4, "web": 2}
    rerank_k: int = 3
    rerank_threshold: float = 0.01
    generations: Annotated[list, operator.add]

from langchain_core.messages import HumanMessage
def refined_question(state:OverAllState):


    print(len(state['questions']))
    print(state['questions'])

    if len(state['questions']) > 0:
        print(1)
        original_question = ""
        for q, a in zip(state['questions'][-4:], state['generations'][-4:]):
            a_without_think = re.sub(r"<think>.*?</think>", "", a, flags=re.DOTALL).strip()
            original_question += f"question: {q}, answer: {a_without_think}"
        original_question += f"final question is {state['question']}"  
    else:
           print(2)
           original_question = state['question']

    prompt = f"""Convert below original question into refined question in Korean for more efficient search and generation.
    You must generate the refined question without any answer or redundant expression.
    <original question>
    {original_question}
    """
    refined_question = llm.invoke([HumanMessage(content=prompt)])
    print(f">>>The Original Question is {original_question} and The Refined Qeustion is {refined_question.content}.")
    return {"question": refined_question.content}

def retrieve_agent(state: OverAllState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, 
        that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state['question']
    top_k_doc = state['top_k']['doc']
    
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"fetch_k": 10, "k":top_k_doc},
        )
    documents = retriever.invoke(question)
    documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]

    print(f"---- Retrieve 문서개수: {len(documents)} -{type(documents[0])}")

    return {"context": [documents]}

from langchain_tavily import TavilySearch
from langchain_core.documents import Document

def websearch_agent(state: OverAllState):
    
    """ Retrieve docs from web search """
    question = state['question']
    top_k_web = state['top_k']['web']

    # Search
    print("---Web Search---")
    tavily_search = TavilySearch(max_results=top_k_web)
    search_docs = tavily_search.invoke(question)
    contents = [f"{d['title']} \n {d['content']}" for d in search_docs['results']]
    metas = [d['url'] for d in search_docs['results']]

    documents = []
    for content, url in zip(contents, metas):
        documents.append({"page_content":content, "metadata":{'url':url}})

    print(f"---- Web Search 문서개수: {len(documents)} - {type(documents[0])}")

    return {"context": [documents]} 

def reranking_agent(state:OverAllState):
    """Rerank retrieved documents"""
    print("---RERANK---")

    question = state['question']
    context = state['context']
    rerank_k = state['rerank_k']
    rerank_threshold = state["rerank_threshold"]
    num_of_agents = len(list(state["top_k"].keys()))

    ## 1차원으로 검색 문서 merged : [[문석검색], [웹검색]] --> [문서검색, 웹검색]
    merged_context = [item for sublist in context[num_of_agents*-1:] for item in sublist]   # 에이전트 개수만큼만 리랭킹 대상 문서로 지정 --> 여기서는 에이전트가 두개이고, 끝에서 2개까지 문서 대상 리랭킹
    print(f"---- merged_context: {len(merged_context)} - {type(merged_context[0])}")
    
    top_scores, documents = reranking(query=question, docs=merged_context, min_score = rerank_threshold, top_k= rerank_k)
    print(f"---- Retrieve Reranking 문서개수: {len(documents)} / top_scores: {top_scores}")
    return {'rerank_context': [documents], 'top_scores': [top_scores], "questions": [state['question']]}





# -----------------------------
# Step 2: Build LangGraph
# -----------------------------

from langgraph.checkpoint.memory import MemorySaver

def search_builder(state):
    rag_builder = StateGraph(state)
    rag_builder.add_node("refined_question", refined_question)
    rag_builder.add_node("websearch_agent", websearch_agent)
    rag_builder.add_node("retrieve_agent", retrieve_agent)
    rag_builder.add_node("reranking_agent", reranking_agent)

    rag_builder.add_edge(START, "refined_question")
    rag_builder.add_edge("refined_question", "websearch_agent")
    rag_builder.add_edge("websearch_agent", "reranking_agent")
    rag_builder.add_edge("refined_question", "retrieve_agent") 
    rag_builder.add_edge("retrieve_agent", "reranking_agent")
    rag_builder.add_edge("reranking_agent", END) 

    memory = MemorySaver()
    graph = rag_builder.compile(checkpointer=memory)

    return graph

search_graph = search_builder(OverAllState)




# -----------------------------
# Step 3: FastAPI Input/Output Models
# -----------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
import uuid

app = FastAPI()

# Define request models
class QuestionRequest(BaseModel):
    question: str
    questions: list
    top_k: Dict = {'doc':3, 'web':2}
    rerank_k: Optional[int] = 3
    rerank_threshold: Optional[float] = 0.01
    session_id: str = None  # 클라이언트가 줄 수도 있고 안 줄 수도 있음
    generations: list

class SearchResponse(BaseModel):
    refined_question: str
    documents: List[Dict]
    scores: List[float]


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: QuestionRequest):
    """
    Endpoint to search documents based on a question.
    Performs retrieval from vector store and web search, then reranks results.
    """
    try:
        # ✅ session_id 자동 생성 또는 재사용
        session_id = request.session_id or str(uuid.uuid4())
        thread_id = f"thread-{session_id}"
        # Initialize state with default values
        state = {
            'question': request.question,
            'top_k': request.top_k,
            'rerank_k': request.rerank_k,
            'rerank_threshold': request.rerank_threshold,
            'questions': [],
            'context': [],
            'rerank_context': [],
            'top_scores': [],
        }
        
        # Execute the search graph
        search_result = await search_graph.ainvoke(state, config={"thread_id": thread_id})
        
        # Prepare response
        documents = []
        for doc in search_result['rerank_context'][0]:
            documents.append({
                "page_content": doc["page_content"],
                "metadata": doc["metadata"]
            })
        scores = [score[0] for score in search_result['top_scores'][0]]
        
        return SearchResponse(
            refined_question = search_result["question"],
            # documents=documents,
            # scores=scores
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Search failed: {str(e)}"
        )

from typing import TypedDict, Annotated, List, Dict, Optional
from pydantic import BaseModel
import re
import json
import asyncio
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import operator

## 현재 일반 어시스턴트 관점에서 작성됨..
SYSTEM_PROMPT = """You are a Smart AI Assistant. 
Use the following pieces of retrieved contexts or web search results to answer the question.
Think step by step and generate logical and reasonable answer in Korean Language.
Nonetheless, If you don't know the answer, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:"""


# class GenState(TypedDict):
#     """
#     Represents the overall state of our graph.
#     """
#     question: str
#     questions: Annotated[List[str], operator.add]
#     documents: Annotated[List[Dict], operator.add]
#     generations: Annotated[List[str], operator.add]

class GenerationRequest(BaseModel):
    question: str
    questions: Annotated[List[str], operator.add]
    documents: Annotated[List[Dict], operator.add]
    session_id: str

class GenerationResponse(BaseModel):
    answer: str

def format_docs(docs: List[Document]) -> str:
    """Format documents for context"""
    return "\n\n".join(doc["page_content"] for doc in docs)

def build_question_history(state: OverAllState) -> str:
    """Build question history from state"""
    if not state.get('generations'):
        return state['question']
    
    history = []
    for q, a in zip(state['questions'][-4:], state['generations'][-4:]):
        a_without_think = re.sub(r"<think>.*?</think>", "", a, flags=re.DOTALL).strip()
        history.append(f"question: {q}, answer: {a_without_think}")
    
    history.append(f"final question is {state['question']}")
    return "\n".join(history)

async def generate_agent(state: OverAllState) -> Dict[str, List[str]]:
    """Generate response based on question and context"""
    question = build_question_history(state)
    context = format_docs(state['context'])
    
    prompt = ChatPromptTemplate.from_messages([("human", SYSTEM_PROMPT),])
    rag_chain = prompt | llm_think | StrOutputParser()
    
    generation = await rag_chain.ainvoke({"context": context, "question": question})
    return {"generations": [generation]}

def create_gen_graph() -> StateGraph:
    """Create and configure the generation graph"""
    rag_builder = StateGraph(OverAllState)
    rag_builder.add_node("generate_agent", generate_agent)
    
    rag_builder.add_edge(START, "generate_agent")
    rag_builder.add_edge("generate_agent", END)
    
    memory = MemorySaver()
    return rag_builder.compile(checkpointer=memory)

gen_graph = create_gen_graph()

def sse_format(payload: Dict) -> str:
    """Format data for SSE"""
    return f"data: {json.dumps(payload,  ensure_ascii=False)}\n\n"

async def generate_stream(state: Dict, thread_id: str):
    """Generate streaming response"""
    thinking_started = False
    
    async for msg, metadata in gen_graph.astream(
        state, 
        stream_mode="messages", 
        config={"thread_id": thread_id}
    ):
        if metadata["langgraph_node"] != "generate_agent":
            continue
            
        if msg.content:
            if thinking_started:
                yield sse_format({"type": "thinking_end"})
                thinking_started = False
                
            yield sse_format({
                "content": msg.content,
                "type": "answer",
                "thinking": False
            })
            
        if "reasoning_content" in msg.additional_kwargs:
            if not thinking_started:
                yield sse_format({"type": "thinking_start"})
                thinking_started = True
                
            yield sse_format({
                "content": msg.additional_kwargs["reasoning_content"],
                "type": "thinking",
                "thinking": True
            })

@app.post("/generate", response_model=GenerationResponse)
async def generate_content(request: GenerationRequest):
    """Endpoint for generating responses"""
    thread_id = f"thread-{request.session_id}"
    
    state = {
        "question": request.question,
        "questions": request.questions,
        "documents": request.documents,
    }
    
    return StreamingResponse(
        generate_stream(state, thread_id),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, workers=1)
