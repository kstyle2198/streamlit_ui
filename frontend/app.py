import base64
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

# CSS 스타일 추가 (hover 시 box-shadow와 transform 적용)
st.markdown("""
    <style>
    a.clickable-box-wrapper {
        display: block;
        text-decoration: none !important;
        color: inherit !important;
    }

    .hover-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        text-decoration: none;
        color: inherit;
        display: block;
        height: 270px;
        overflow: hidden;
    }

    .hover-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        cursor: pointer;
    }

    /* 텍스트 서식 Streamlit에 맞춤 */
    .hover-box h3, .hover-box p {
        all: unset;
        display: block;
    }

    .hover-box h3 {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .hover-box p {
        font-size: 1rem;
        color: inherit;
    }

    </style>
""", unsafe_allow_html=True)


def make_hover_container(title:str, content:str, url:str):
    st.markdown(f"""
            <a href="{url}" target="_blank" class="clickable-box-wrapper">
            <div class="hover-box">
                <h1>{title}</h1>
                <p>{content}</p></div>
            </a>
        """, unsafe_allow_html=True)
    
# def make_hover_image_from_file(image_file, max_width: str = "100%"):
#     if isinstance(image_file, str):
#         with open(image_file, "rb") as f:
#             img_bytes = f.read()
#     else:
#         img_bytes = image_file.read()

#     img_base64 = base64.b64encode(img_bytes).decode()

#     st.markdown(f"""
#         <style>
#         .hover-image {{
#             transition: transform 0.3s ease;
#             border-radius: 10px;
#             cursor: default;
#             display: block;
#             margin: auto;
#             max-width: {max_width};
#             height: auto;
#         }}
#         .hover-image:hover {{
#             transform: scale(1.05);
#             box-shadow: 0 4px 20px rgba(0,0,0,0.3);
#         }}
#         </style>
#         <img src="data:image/jpeg;base64,{img_base64}" class="hover-image" />
#     """, unsafe_allow_html=True)


image_paths = [
    "./system_image/img1.jpg",
    "./system_image/img2.jpg",
    "./system_image/img3.jpg",
    "./system_image/img4.jpg",
]
# base64로 인코딩된 이미지 태그 생성 함수
def get_base64_img_tag(file_path):
    with open(file_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        return f'<img src="data:image/png;base64,{encoded}" style="width: 100%; position: absolute; opacity: 0; transition: opacity 1s;">'

# 이미지 태그 리스트 생성
image_tags = ''.join([get_base64_img_tag(path) for path in image_paths])

# HTML + JS 코드로 슬라이드쇼 구성
html_code = f"""
<div id="slideshow" style="position: relative; width: 100%; max-width: 800px; margin: auto; height: 500px;">
  {image_tags}
</div>

<script>
const slides = document.querySelectorAll("#slideshow img");
let current = 0;

function showNextSlide() {{
    slides[current].style.opacity = 0;
    current = (current + 1) % slides.length;
    slides[current].style.opacity = 1;
}}

slides[0].style.opacity = 1;
setInterval(showNextSlide, 3000);
</script>
"""

def make_home():
    with st.container():
        col11, col12, col13 = st.columns([3, 0.2, 7])
        with col11: 
            with st.container(height=200): 
                st.info("제목 공간")
            components.html(html_code, height=400)
            with st.container(height=300): 
                st.info("Update News")

        with col12: pass 
        with col13: 
            col111, col112, col113 = st.columns(3)

            temp_content1 = """ <h3 style="color: grey;">MOM or PCF Search and Tech Review</h3>
            <h7 style="color: gray;">- Keyword(Single/Multiple) or Hybrid Search</h7><br>
            <h7 style="color: gray;">- AgGrid Table Display</h7><br>
            <h7 style="color: gray;">- Agent Technical Review</h7><br>
            """

            with col111: 
                make_hover_container(title="🌻 Keyword Search", content=temp_content1, url="http://localhost:8501/KeywordSearch")
            with col112: 
                make_hover_container(title="Hybrid Search", content="I like an apple", url="http://localhost:8501/HybridSearch")
            with col113: 
                make_hover_container(title="Rag Agent", content="I like an apple", url="http://localhost:8501/RagAgent")
            
            st.markdown("---")
            col121, col122, col123 = st.columns(3)
            with col121: 
                make_hover_container(title="Deep Research", content="I like an apple", url="http://localhost:8501/DeepResearch")
            with col122: 
                make_hover_container(title="LightRag with Knowledge Graph", content="I like an apple", url="http://localhost:8501/LightRAg")
            with col123: 
                make_hover_container(title="Project File Uploader", content="I like an apple", url="http://localhost:8501/File_Uploader")

            st.markdown("---")
            col131, col132, col133 = st.columns(3)
            with col131: 
                make_hover_container(title="Agent_01", content="I like an apple", url="")
            with col132: 
                make_hover_container(title="Agent_02", content="I like an apple", url="")
            with col133: 
                make_hover_container(title="Agent_03", content="I like an apple", url="")

def side_bar():
    with st.sidebar:
        st.markdown("Sidebar")



if __name__ == "__main__":

    side_bar()

    make_home()

    