from langchain_community.tools import TavilySearchResults
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv(override=True)

def do_web_search(question:str):

    web_search_tool1 = TavilySearchResults(
        max_results=3,
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        # search_depth="advanced",
        # include_domains = []
        # exclude_domains = []
        )
    web_results = web_search_tool1.invoke({'query': question})
    return web_results


from pydantic import BaseModel
class SearchRequest(BaseModel):
    query: str


from fastapi import APIRouter, HTTPException
web_search = APIRouter(prefix="/web_search")

@web_search.post("/", tags=["Search"])
def do_search(request: SearchRequest):
    try:
        results = do_web_search(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



