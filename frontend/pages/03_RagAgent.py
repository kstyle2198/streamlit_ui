import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed"
)

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


def main():
    # 오른쪽 패널 생성
    col1, col2 = st.columns([3, 1])  # 3:1 비율로 좌우 분할

    # 메인 콘텐츠 영역
    with col1:
        # Initialize chat history with docs and paths if not exists
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Ensure all assistant messages have docs and paths
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                if "docs" not in msg:
                    msg["docs"] = []
                if "paths" not in msg:
                    msg["paths"] = []

        # Custom function to render message content
        def render_message_content(message):
            content = message["content"]
            
            # Assistant 메시지 처리
            if message["role"] == "assistant":
                # Thinking Process가 있는 경우 처리
                if "🤔 Thinking Process:" in content:
                    parts = content.split("\n\n---\n\n")
                    if len(parts) == 2:
                        think_content = parts[0].replace("🤔 Thinking Process:\n", "")
                        response_content = parts[1]
                        
                        with st.expander("🤔 Thinking Process", expanded=False):
                            st.markdown(think_content)
                        st.markdown(response_content)
                        
                        # docs와 paths 정보 표시 (항상 표시)
                        if "docs" in message and message["docs"]:
                            st.info(f"참조 문서: {message['docs']}")
                        if "paths" in message and message["paths"]:
                            st.info(f"파일 경로: {message['paths']}")
                        return
            
                # 일반 assistant 메시지 처리
                st.markdown(content)
                # docs와 paths 정보 표시 (항상 표시)
                if "docs" in message and message["docs"]:
                    st.info(f"참조 문서: {message['docs']}")
                if "paths" in message and message["paths"]:
                    st.info(f"파일 경로: {message['paths']}")
                    for img in message['paths']:
                        st.markdown(img)
                        st.image(image=img)
                    st.rerun()
            else:
                # User 메시지 처리
                st.markdown(content)

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                render_message_content(message)

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
        
        # Display user message immediately
        with col1:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Prepare assistant response
        with col1:
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
                    "style": joke_style
                }
                
                # Make the POST request with streaming
                try:
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
                except requests.exceptions.RequestException as e:
                    st.error(f"서버 연결 오류: {e}")
                    full_response = "서버에 연결할 수 없습니다."
                
                # Final render without cursor
                think_container.markdown(think_content)
                response_container.markdown(full_response)
                
                # Add to chat history with special formatting and metadata
                if think_content:
                    formatted_content = f"🤔 Thinking Process:\n{think_content}\n\n---\n\n{full_response}"
                else:
                    formatted_content = full_response
                    
                # 실제 애플리케이션에서는 여기서 docs와 paths를 서버 응답에서 추출해야 함
                # 예제를 위해 하드코딩된 값 사용
                assistant_message = {
                    "role": "assistant", 
                    "content": formatted_content, 
                    "docs": ["doc1", "doc2"],  # 서버 응답에서 이 데이터를 가져와야 함
                    "paths": ["D:/Streamlit_UI/frontend/static/images/test_image.jpg", "D:/Streamlit_UI/frontend/static/images/test_image.jpg"]  # 서버 응답에서 이 데이터를 가져와야 함
                }
                
                st.session_state.messages.append(assistant_message)
                
                # 새 메시지의 docs와 paths도 표시
                if "docs" in assistant_message and assistant_message["docs"]:
                    st.info(f"참조 문서: {assistant_message['docs']}")
                if "paths" in assistant_message and assistant_message["paths"]:
                    st.info(f"파일 경로: {assistant_message['paths']}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("Rag Agent")
    main()
