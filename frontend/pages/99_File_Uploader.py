from utils import FileUploader
import streamlit as st
# st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

import sys
sys.path.append("../")



file_uploader = FileUploader()

if __name__ == "__main__":
    st.title("Project File Uploader")
    project_name = st.text_input("PROJECT NAME")
    with st.container():
        file_uploader.file_uploader(project_name=project_name)

    