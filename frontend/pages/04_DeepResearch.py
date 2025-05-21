import streamlit as st
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")


import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# 예제 데이터
df = pd.DataFrame({
    "comment": [
        "<p style='color:red;'>We request the installation of a <b>hybrid SOx scrubber</b> system instead of the open-loop type.</p>",
        "<p>This is <i>italic</i> and <u>underlined</u>.</p>"
    ]
})

# HTML을 innerHTML로 설정하는 렌더러
html_renderer = JsCode("""
function(params) {
    let span = document.createElement('span');
    span.innerHTML = params.value;
    return span;
}
""")

# Grid 옵션 설정
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_column("comment", cellRenderer=html_renderer)
grid_options = gb.build()

# AgGrid 렌더링
AgGrid(
    df,
    gridOptions=grid_options,
    allow_unsafe_jscode=True,
    enable_enterprise_modules=False,
    fit_columns_on_grid_load=True
)



if __name__ == "__main__":
    st.title("Deep Research")