import streamlit as st
import pandas as pd
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd

# 샘플 데이터
data = {
    "상태": ["<span style='color:green;font-weight:bold'>정상</span>", 
            "<span style='color:orange'>주의</span>", 
            "<span style='color:red;font-weight:bold'>위험</span>"],
    "설명": ["시스템 정상 작동", "일부 지연 발생", "심각한 오류 발생"],
    "수치": [25, 65, 92]
}
df = pd.DataFrame(data)

# Grid 설정
gb = GridOptionsBuilder.from_dataframe(df)

# 상태 컬럼에 커스텀 렌더러 적용
gb.configure_column(
    "상태",
    cellRenderer="""
    function(params) {
        return `<div style="padding: 5px; border-radius: 5px;">${params.value}</div>`;
    }
    """
)

# 수치 컬럼에 조건부 서식 적용
gb.configure_column(
    "수치",
    cellRenderer="""
    function(params) {
        let color = 'black';
        if (params.value > 90) color = 'red';
        else if (params.value > 60) color = 'orange';
        return `<div style="color:${color}; font-weight:bold">${params.value}</div>`;
    }
    """
)

grid_options = gb.build()

# AgGrid 표시
AgGrid(
    df,
    gridOptions=grid_options,
    height=200,
    allow_unsafe_jscode=True,
    escapeHtml=False,
    fit_columns_on_grid_load=True
)

# if __name__ == "__main__":
#     st.title("Hybrid Search")