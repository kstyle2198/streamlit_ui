import streamlit as st
import requests
import json
import sseclient
import time

# Page config
st.set_page_config(
    page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed"
)

import streamlit as st
import requests
import sseclient
import json

import streamlit as st
import requests
import json

# CSS를 이용한 하단 고정 채팅 입력창
st.markdown("""
<style>
    /* 채팅 입력창 고정 */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        z-index: 100;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
    }
    
    /* columns 레이아웃 조정 */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* 오른쪽 패널 조정 */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
        padding-bottom: 120px;
    }
</style>
""", unsafe_allow_html=True)


# 오른쪽 패널 생성
col1, col2 = st.columns([3, 1])  # 3:1 비율로 좌우 분할

# 메인 콘텐츠 영역
with col1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Custom function to render message content
    def render_message_content(content):
        if "🤔 Thinking Process:" in content:
            parts = content.split("\n\n---\n\n")
            if len(parts) == 2:
                think_content = parts[0].replace("🤔 Thinking Process:\n", "")
                response_content = parts[1]
                
                with st.expander("🤔 Thinking Process", expanded=False):
                    st.markdown(think_content)
                st.markdown(response_content)
                return
        
        st.markdown(content)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_message_content(message["content"])

# 오른쪽 패널 (사이드바 대체)
with col2:
    st.title("설정 패널")
    st.markdown("""
    ### 앱 정보
    이 앱은 사용자가 입력한 주제에 따라 농담을 생성합니다.
    """)
    
    # 설정 옵션 예시
    joke_style = st.selectbox(
        "농담 스타일",
        ("유머러스", "아이스크림", "블랙코미디", "펀"),
        key="joke_style"
    )
    
    st.divider()
    st.markdown("""
    ### 추가 기능
    - 설정 1
    - 설정 2
    - 설정 3
    """)
    
    if st.button("채팅 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 하단 고정 채팅 입력창
st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
if prompt := st.chat_input("Enter a topic for your joke"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col1:  # 메시지는 메인 컬럼에 표시
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Prepare assistant response
    with col1:  # 응답도 메인 컬럼에 표시
        with st.chat_message("assistant"):
            # Create containers
            think_expander = st.expander("🤔 Thinking Process", expanded=True)
            think_container = think_expander.empty()
            response_container = st.empty()
            
            full_response = ""
            think_content = ""
            in_think_block = False
            
            # Prepare the request
            url = "http://localhost:8000/generate"
            headers = {'Accept': 'text/event-stream'}
            data = {
                "topic": prompt,
                "style": joke_style  # 사이드바에서 선택한 옵션 사용
            }
            
            # Make the POST request with streaming
            with requests.post(url, json=data, headers=headers, stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data:'):
                            try:
                                data = json.loads(decoded_line[5:].strip())
                                content = data.get("content", "")
                                
                                # Think block processing
                                if "<think>" in content:
                                    in_think_block = True
                                    content = content.replace("<think>", "")
                                
                                if "</think>" in content:
                                    in_think_block = False
                                    content = content.replace("</think>", "")
                                
                                if in_think_block or "<think>" in content or "</think>" in content:
                                    think_content += content
                                    think_container.markdown(think_content + "▌")
                                else:
                                    full_response += content
                                    response_container.markdown(full_response + "▌")
                                
                            except json.JSONDecodeError:
                                st.error("Error decoding server response")
            
            # Final render without cursor
            think_container.markdown(think_content)
            response_container.markdown(full_response)
            
            # Add to chat history with special formatting
            if think_content:
                formatted_content = f"🤔 Thinking Process:\n{think_content}\n\n---\n\n{full_response}"
            else:
                formatted_content = full_response
                
            st.session_state.messages.append({"role": "assistant", "content": formatted_content})

st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass
    # st.title("Rag Agent")
