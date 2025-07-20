import re
from typing import List, Dict, Optional, Union, TypedDict, Annotated
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from FlagEmbedding import FlagReranker
import operator
import heapq
import json
import asyncio
import uuid
from fastapi.responses import StreamingResponse

from dotenv import load_dotenv
load_dotenv(override=True)

llm = ChatGroq(model_name="gemma2-9b-it", temperature=0,max_tokens=3000,) 
llm_think = ChatGroq(model_name="qwen/qwen3-32b", temperature=0,max_tokens=3000,) 

reranking_model_path = "D:/LLMs/bge-reranker-v2-m3" # 실제 경로로 변경해야 합니다.
reranker = FlagReranker(model_name_or_path=reranking_model_path, 
                        use_fp16=True,
                        batch_size=512,
                        max_length=2048,
                        normalize=True)

from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from langchain_core.documents import Document

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

def reranking(query: str, docs: list, min_score: float = 0.5, top_k: int = 3):
    """
    Reranks documents based on a query.
    """
    global reranker
    inputs = [[query.lower(), doc["page_content"].lower()] for doc in docs]
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


### REFINE QUERY ######################
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


class RefineState(TypedDict):
    """
    """
    question: str
    chat_history: list
    refined_query: str
    

def format_query(docs: list) -> str:
    """Format documents for context"""
    try:
        return "\n\n".join(doc["content"].strip().replace("\n", "") for doc in docs)
    except:
        return ""


from langchain_core.messages import HumanMessage
def refined_question(state:RefineState):
    question = state['question']
    chat_history = state['chat_history']
    format_chat_history = format_query(chat_history)

    prompt = f"""아래 질문 내용이 chat_history와 관련이 있는 경우, chat_history를 참고하여 질문을 정제해 주세요.
    chat_history의 내용이 없거나, 내용이 질문과 상관이 없는 경우, 질문 자체의 표현만 정제해주세요.
    질문에 대해 답하지 말고, 표현만 정제해주세요.
    <question>
    {question}

    <chat_history>
    {format_chat_history}
    """
    refined_question_content = llm.invoke([HumanMessage(content=prompt)]).content
    print(f">>>The Original Question is {question}")
    print(f">>>The Refined Question is {refined_question_content}.")
    return {"refined_query": refined_question_content}


from langgraph.checkpoint.memory import MemorySaver

def query_refiner(state):
    refine_builder = StateGraph(state)
    refine_builder.add_node("refine_agent", refined_question)

    refine_builder.add_edge(START, "refine_agent")
    refine_builder.add_edge("refine_agent", END) 
    return refine_builder.compile()

refine_graph = query_refiner(RefineState)


app = FastAPI()

class RefineRequest(BaseModel):
    question: str
    chat_history: list = []

class RefineResponse(BaseModel):
    refined_query: str

@app.post("/refine", response_model=RefineResponse)
async def refine_question(req: RefineRequest):
    try:
        # 초기 상태 구성
        state = {
            "question": req.question,
            "chat_history": req.chat_history
            }

        # 그래프 실행
        result = refine_graph.invoke(state)
        return RefineResponse(refined_query=result["refined_query"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



### HYBRID SEARCH ######################
class OverAllState(TypedDict):
    """
    Represents the overall state of our graph.
    """
    question: str
    questions: Annotated[list, operator.add] # 질문 누적
    context: Annotated[list, operator.add] # 검색 결과 누적
    rerank_context: Annotated[list, operator.add] # 리랭크된 검색 결과 누적
    top_scores: Annotated[list, operator.add] # 리랭크 점수 누적
    top_k: Dict = {'doc':4, "web": 2}
    rerank_k: int = 3
    rerank_threshold: float = 0.01
    generations: Annotated[list, operator.add] # 응답 결과 누적



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

    print(f"---- Retrieve 문서개수: {len(documents)}")

    # 검색 결과를 context에 추가
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

    print(f"---- Web Search 문서개수: {len(documents)}")

    # 웹 검색 결과를 context에 추가
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
    # 에이전트 개수만큼만 리랭킹 대상 문서로 지정 (여기서는 에이전트가 두개이고, 끝에서 2개까지 문서 대상 리랭킹)
    merged_context = [item for sublist in context[num_of_agents*-1:] for item in sublist] 
    print(f"---- merged_context: {len(merged_context)}")
    
    top_scores, documents = reranking(query=question, docs=merged_context, min_score = rerank_threshold, top_k= rerank_k)
    print(f"---- Reranking 문서개수: {len(documents)} / top_scores: {top_scores}")
    
    # 리랭크된 문서와 스코어를 각각의 리스트에 추가
    return {"rerank_context": [documents], "top_scores": [top_scores]}


def search_builder(state):
    rag_builder = StateGraph(state)
    rag_builder.add_node("websearch_agent", websearch_agent)
    rag_builder.add_node("retrieve_agent", retrieve_agent)
    rag_builder.add_node("reranking_agent", reranking_agent)

    rag_builder.add_edge(START, "websearch_agent")
    rag_builder.add_edge("websearch_agent", "reranking_agent")
    rag_builder.add_edge(START, "retrieve_agent") 
    rag_builder.add_edge("retrieve_agent", "reranking_agent")
    rag_builder.add_edge("reranking_agent", END) 

    memory = MemorySaver()
    return rag_builder.compile(checkpointer=memory)


search_graph = search_builder(OverAllState)


# Define request models
class QuestionRequest(BaseModel):
    question: str
    top_k: Dict = {'doc':3, 'web':2}
    rerank_k: Optional[int] = 3
    rerank_threshold: Optional[float] = 0.01
    session_id: str = None  # 클라이언트가 줄 수도 있고 안 줄 수도 있음

class SearchResponse(BaseModel):
    refined_question: str
    documents: List[Dict]
    scores: List[float]
    questions_history: List[str] # 누적된 질문 히스토리 추가
    search_results_history: List[List[Dict]] # 누적된 검색 결과 히스토리 추가 (각 검색 단계별 결과 리스트)
    reranked_results_history: List[List[Dict]] # 누적된 리랭크 결과 히스토리 추가 (각 리랭크 단계별 결과 리스트)
    rerank_scores_history: List[List[float]] # 누적된 리랭크 스코어 히스토리 추가 (각 리랭크 단계별 스코어 리스트)

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
        
        # Initialize state with default values and empty lists for accumulation
        state = {
            "question": request.question,
            "top_k": request.top_k,
            "rerank_k": request.rerank_k,
            "rerank_threshold": request.rerank_threshold,
            "questions": [], # 여기에 첫 질문이 refined_question에서 추가됨
            "context": [],
            "rerank_context": [],
            "top_scores": [],
            "generations": [], # /search에서는 사용하지 않지만, state 정의에 포함
        }
        
        # Execute the search graph
        # ainvoke는 최종 상태를 반환합니다.
        search_result = await search_graph.ainvoke(state, config={"thread_id": thread_id})
        
        # Prepare response
        documents = []
        # rerank_context는 리스트의 리스트이므로, 가장 마지막 리랭크 결과를 가져옵니다.
        if search_result["rerank_context"]:
            documents = search_result["rerank_context"][-1] 

        scores = []
        # top_scores도 리스트의 리스트이므로, 가장 마지막 리랭크 스코어를 가져옵니다.
        if search_result["top_scores"]:
            scores = [score[0] for score in search_result["top_scores"][-1]]
        
        return SearchResponse(
            refined_question = search_result["question"],
            documents=documents,
            scores=scores,
            questions_history=search_result["questions"], # 누적된 질문 히스토리
            search_results_history=search_result["context"], # 누적된 검색 결과 히스토리
            reranked_results_history=search_result["rerank_context"], # 누적된 리랭크 결과 히스토리
            rerank_scores_history=[[s[0] for s in score_list] for score_list in search_result["top_scores"]] # 누적된 리랭크 스코어 히스토리
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Search failed: {str(e)}"
        )


### GENERATE ########################
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
    
    async for msg, metadata in gen_graph.astream(state, stream_mode="messages", config={"thread_id": thread_id}):
        if metadata["langgraph_node"] != "generate_agent":
            continue

        if msg.content == "<think>":
            yield sse_format({
                "content": msg.content,
                "type": "thinking_start",
                "thinking": True
                })
            thinking_started = True
        elif msg.content == "</think>":
            yield sse_format({
                "content": msg.content,
                "type": "thinking_end",
                "thinking": False
                })
            thinking_started = False
        elif msg.content and thinking_started:
            yield sse_format({
                "content": msg.content,
                "type": "thinking",
                "thinking": True
                })
        elif msg.content and not thinking_started:
            yield sse_format({
                "content": msg.content,
                "type": "answer",
                "thinking": False
                })
        else: pass

@app.post("/generate", response_model=GenerationResponse)
async def generate_content(request: GenerationRequest):
    """Endpoint for generating responses"""
    thread_id = f"thread-{request.session_id}"
    
    state = {
        "question": request.question,
        "questions": request.questions,
        "documents": request.documents,
        }
    
    return StreamingResponse(generate_stream(state, thread_id), media_type="text/event-stream")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, workers=1)
