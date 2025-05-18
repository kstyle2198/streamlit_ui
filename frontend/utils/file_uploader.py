import streamlit as st
import os
from pathlib import Path
parent_dir = Path(__file__).parent.parent
base_dir = str(parent_dir) + "\data"

@st.fragment
class FileUploader():
    def __init__(self):
        pass
    def file_uploader(self, project_name):
        uploaded_file = st.file_uploader("📎Upload your file")
        if uploaded_file:
            temp_dir = base_dir+f"\{project_name}"   # tempfile.mkdtemp()  --->  import tempfile 필요, 임시저장디렉토리 자동지정함
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                    f.write(uploaded_file.getvalue())
        
        if st.button("Save", type='secondary'):
            st.markdown(f"path: {path}")
            st.info("Saving a file is completed")
        else:
            st.empty()

        

