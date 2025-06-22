import streamlit as st
import requests
from typing import List, Dict
import uuid
import json

# Configuration
BACKEND_URL = "http://localhost:8000"  # Update with your FastAPI server URL

# Initialize session state
def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "thinking" not in st.session_state:
        st.session_state.thinking = False

initialize_session_state()

# UI Layout
st.title("AI Assistant with RAG")

# Sidebar for settings and new conversation button
with st.sidebar:
    st.header("Settings")
    
    # New conversation button
    if st.button("🔄 New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.session_state.search_results = []
        st.session_state.thinking = False
        st.rerun()
    
    st.header("Search Settings")
    doc_top_k = st.slider("Document Top K", 1, 10, 3)
    web_top_k = st.slider("Web Search Top K", 1, 5, 2)
    rerank_k = st.slider("Rerank K", 1, 5, 3)
    rerank_threshold = st.slider("Rerank Threshold", 0.0, 1.0, 0.01)

# Display conversation history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("thinking"):
            with st.expander("AI Thinking Process"):
                st.markdown(message["thinking"])
        
        # Show sources only for the most recent assistant message
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("References"):
                for i, source in enumerate(message["sources"]):
                    st.subheader(f"Source {i+1}")
                    st.markdown(source["content"])
                    if source.get("url"):
                        st.markdown(f"[View original]({source['url']})")

# User input
if prompt := st.chat_input("Ask me anything"):
    # Initialize new conversation state if it's the first message
    if not st.session_state.conversation_history:
        st.session_state.search_results = []
    
    st.session_state.current_question = prompt
    
    # Add user message to chat
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.conversation_history.append({
        "role": "user",
        "content": prompt
    })

    # Step 1: Perform search
    search_payload = {
        "question": prompt,
        "top_k": {"doc": doc_top_k, "web": web_top_k},
        "rerank_k": rerank_k,
        "rerank_threshold": rerank_threshold,
        "session_id": st.session_state.session_id
    }
    
    try:
        search_response = requests.post(
            f"{BACKEND_URL}/search",
            json=search_payload
        )
        search_response.raise_for_status()
        search_data = search_response.json()
        
        # Store search results for this question only
        current_search_results = search_data["documents"]
        
        # Show refined question if different from original
        if search_data["refined_question"] != prompt:
            with st.chat_message("assistant"):
                st.markdown(f"🔍 Refined question: *{search_data['refined_question']}*")
        
        # Step 2: Generate answer with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            thinking_content = ""
            
            # Prepare generation request
            gen_payload = {
                "question": search_data["refined_question"],
                "questions": [msg["content"] for msg in st.session_state.conversation_history if msg["role"] == "user"],
                "documents": current_search_results,
                "session_id": st.session_state.session_id
            }
            
            # Stream the response
            with requests.post(
                f"{BACKEND_URL}/generate",
                json=gen_payload,
                stream=True
            ) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            try:
                                data = json.loads(decoded_line[5:].strip())
                                
                                if data.get("type") == "thinking_start":
                                    st.session_state.thinking = True
                                    thinking_content = ""
                                    with message_placeholder.container():
                                        st.markdown(full_response)
                                        with st.expander("AI is thinking..."):
                                            thinking_placeholder = st.empty()
                                
                                elif data.get("type") == "thinking_end":
                                    st.session_state.thinking = False
                                
                                elif data.get("type") == "thinking":
                                    thinking_content += data["content"]
                                    if st.session_state.thinking:
                                        with message_placeholder.container():
                                            st.markdown(full_response)
                                            with st.expander("AI is thinking..."):
                                                st.markdown(thinking_content)
                                
                                elif data.get("type") == "answer":
                                    full_response += data["content"]
                                    message_placeholder.markdown(full_response + "▌")
                            
                            except json.JSONDecodeError:
                                continue
            
            message_placeholder.markdown(full_response)
            
            # Format sources for display
            sources = []
            for doc in current_search_results:
                source = {
                    "content": doc["page_content"],
                    "url": doc["metadata"].get("url") if "metadata" in doc else None
                }
                sources.append(source)
            
            # Add to conversation history with sources
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": full_response,
                "thinking": thinking_content if thinking_content else None,
                "sources": sources
            })
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the backend: {str(e)}")

# Debug section (optional)
with st.expander("Debug Info"):
    st.write("Session ID:", st.session_state.session_id)
    st.write("Current Search Results:", st.session_state.search_results)
    st.write("Conversation History:", st.session_state.conversation_history)