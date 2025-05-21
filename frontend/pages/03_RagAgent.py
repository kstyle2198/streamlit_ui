import streamlit as st
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
st.session_state

# Sample data
# 초기화: 세션 상태에 df 저장
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Comment": ["Hello", "Nice to meet you", "Welcome"]
    })

df = st.session_state.df  # 항상 세션 상태의 df 사용

st.title("AgGrid 셀 클릭 -> 텍스트 편집 팝업 예제")

# 세션 상태 초기화
if "show_popup" not in st.session_state:
    st.session_state["show_popup"] = False
if "selected_cell_value" not in st.session_state:
    st.session_state["selected_cell_value"] = ""
if "selected_row_index" not in st.session_state:
    st.session_state["selected_row_index"] = -1

# AgGrid 옵션 구성
gb = GridOptionsBuilder.from_dataframe(df)
gb.configure_selection(selection_mode="single", use_checkbox=False)
grid_options = gb.build()

grid_response = AgGrid(
    df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    height=300,
    theme="alpine",
)

selected = grid_response["selected_rows"]

# 셀 클릭 처리
try:
    if selected.shape[0] > 0:
        st.session_state["selected_cell_value"] = selected["Comment"].values[0]
        st.session_state["selected_row_index"] = df[df["ID"] == selected["ID"].values[0]].index[0]
        st.session_state["show_popup"] = True
        st.rerun()

except: pass

@st.dialog("내용 수정")
def cell_modify():            
    updated_comment = st.text_area("Edit Comment", st.session_state["selected_cell_value"])
    if st.button("💾 저장"):
        df.at[st.session_state["selected_row_index"], "Comment"] = updated_comment
        st.session_state["show_popup"] = False
        st.rerun()

# 팝업 텍스트 입력창 보여주기
if st.session_state["show_popup"]:
    cell_modify()



if __name__ == "__main__":
    st.title("Rag Agent")