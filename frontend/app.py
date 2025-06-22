import base64
import pathlib
import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")


# Function to load CSS from the 'assets' folder
def load_css(file_path):
    with open(file_path, encoding="utf8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the external CSS
css_path = pathlib.Path("D:/Streamlit_UI/frontend/assets/style.css")
load_css(css_path)

page_home = st.Page("pages/00_HomePage.py", title="Home Page", default=True)
page_keyword = st.Page("pages/01_KeywordSearch.py", title="Keyword Search", default=False)
page_hybridsearch = st.Page("pages/02_RagAgent_Multi.py", title="Rag Agent - Multi Turn", default=False)
page_rag = st.Page("pages/03_RagAgent.py", title="Rag Agent", default=False)

pg = st.navigation({"Set1": [page_home, page_keyword, page_hybridsearch, page_rag]}, position="sidebar")
pg.run()


