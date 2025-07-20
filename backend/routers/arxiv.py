from langchain_community.utilities import ArxivAPIWrapper
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


def parse_paper_info(text: str) -> dict:
    lines = text.strip().split('\n')
    result = {}
    current_key = None
    summary_lines = []

    for line in lines:
        if line.startswith('Published:'):
            result['Published'] = line[len('Published:'):].strip()
            current_key = None
        elif line.startswith('Title:'):
            result['Title'] = line[len('Title:'):].strip()
            current_key = None
        elif line.startswith('Authors:'):
            result['Authors'] = line[len('Authors:'):].strip()
            current_key = None
        elif line.startswith('Summary:'):
            current_key = 'Summary'
            summary_lines.append(line[len('Summary:'):].strip())
        elif current_key == 'Summary':
            summary_lines.append(line.strip())

    if summary_lines:
        result['Summary'] = ' '.join(summary_lines)

    return result


def do_arxiv_search(question:str):

    arxiv = ArxivAPIWrapper(top_k_results=5)
    arxiv_results = arxiv.run(question)
    arxiv_results = arxiv_results.split("\n\n")    
    refined_results = []
    for d in arxiv_results:
        refined_d = parse_paper_info(d)
        refined_results.append(refined_d)
    return refined_results


from pydantic import BaseModel
class SearchRequest(BaseModel):
    query: str


from fastapi import APIRouter, HTTPException
arxiv_search = APIRouter(prefix="/arxiv_search")

@arxiv_search.post("/", tags=["Search"])
def do_search(request: SearchRequest):
    try:
        results = do_arxiv_search(request.query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



