import streamlit as st
import pandas as pd
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import markdown

# 샘플 데이터
data = {
    "제목": ["**강조** 텍스트", "*기울임* 텍스트", "`코드` 블록"],
    "내용": [
        "이것은 [링크](https://example.com)입니다.",
        "리스트:\n- 항목1\n- 항목2",
        "```python\nprint('Hello World')\n```"
        "~~I like an apple~~"
    ]
}

# Markdown을 HTML로 변환
for i in range(len(data["내용"])):
    data["내용"][i] = markdown.markdown(data["내용"][i])

# AgGrid 설정
gb = GridOptionsBuilder.from_dataframe(pd.DataFrame(data))
gb.configure_default_column(
    autoHeight=True,
    wrapText=True,
    cellStyle={"white-space": "normal"}
)

grid_options = gb.build()

# AgGrid 표시
AgGrid(pd.DataFrame(data), gridOptions=grid_options, allow_unsafe_html=True)

if __name__ == "__main__":
    st.title("Hybrid Search")