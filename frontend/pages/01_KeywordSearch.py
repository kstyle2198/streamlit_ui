import io
import re
import requests
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import streamlit as st
import pandas as pd
from openpyxl import load_workbook
st.set_page_config(page_title="UI", page_icon="🐬", layout="wide", initial_sidebar_state="collapsed")

import streamlit.components.v1 as components


input_template = """You are a smart shipyard engineer.
The following context contains similar historical cases relevant to the Buyer's New Comment. 
It includes records of buyer comments, builder replies, extra, credit, remarks.
Your task is to generate a step-by-step technical review report for the Buyer's New Comment provided below, referencing the context.

Focus primarily on:
- Reviewing Buyer's New Comment by referring the context.
- Distinguish between the buyer's commment and provided context that is similar case in the past.
- Assessing whether the issue is technically acceptable or not.
- Evaluating potential risks to the construction schedule and additional costs.
- Use bullet to specify important information
- You Must Generate the answer based on the context. If there is no relevant information in the conxtext. Just say 'I don't know'.

<<Reference Information>>
- 'Extra' refers to costs borne by the builder.
- 'Credit' refers to costs borne by the buyer.

<<Buyer's New Comment>>
"""

example_input = "request for installing bwts - electric type"

# 초기 데이터
data = pd.read_excel("./data/clari_sample.xlsx")

# 세션 상태 초기화
if 'df' not in st.session_state:
    st.session_state.df = data
if "show_popup" not in st.session_state:
    st.session_state["show_popup"] = False
if "selected_cell_value" not in st.session_state:
    st.session_state["selected_cell_value"] = ""
if "selected_row_index" not in st.session_state:
    st.session_state["selected_row_index"] = -1


# AgGrid 사용자 정의 CSS
custom_css = {
    ".ag-cell": {"font-size": "18px"},
    ".ag-header-cell-label": {"font-size": "18px", "font-weight": "bold"},
    ".ag-header": {
        "background-color": "#FEFFBE",  # 초록색 배경
        "color": "black",
        "font-weight": "bold",
        "font-size": "16px"
    }
}

# text_area 사용자 정의 CSS
st.markdown(
    """
    <style>
    /* 전체 textarea에 적용 */
    textarea {
        font-family: 'Nanum Gothic', sans-serif !important;
        font-size: 18px !important;
        background-color: #f0f4f8 !important;  /* 연한 파스텔 블루 배경 */
        color: #333333 !important;              /* 글자색 진한 회색 */
        border: 1px solid #4a90e2 !important;   /* 파란색 테두리 */
        border-radius: 6px !important;          /* 모서리 둥글게 */
        padding: 20px !important;                /* 내부 여백 */
        box-shadow: 2px 2px 8px rgba(74, 144, 226, 0.3) !important; /* 은은한 그림자 */
        resize: vertical !important;             /* 세로 방향으로만 크기 조절 가능 */
        transition: border-color 0.3s ease;
    }
    textarea:focus {
        border-color: #357ABD !important;        /* 포커스 시 테두리 색상 변경 */
        outline: none !important;                 /* 기본 포커스 외곽선 제거 */
        box-shadow: 0 0 10px rgba(53, 122, 189, 0.5) !important;
        background-color: #e6f0fa !important;    /* 포커스 시 배경 밝게 */
    }
    </style>
    """,
    unsafe_allow_html=True
)


def download_excel(df):
    try:
        st.dataframe(data=df, use_container_width=True)

        # 엑셀로 변환
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()
        data = output.getvalue()

        # 다운로드 버튼
        st.download_button(
            label="Excel 다운로드",
            data=data,
            file_name='data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        st.error("선택된 데이터가 없습니다.")


def text_area_copy(text):
    components.html(f"""
        <div>
            <textarea id="copyTarget" style="
                position: absolute;
                left: -9999px;
                top: 0;
            ">{text}</textarea>
            <button onclick="copyToClipboard()" style="
                padding: 6px 15px;
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            ">Copy</button>
        </div>

        <script>
            function copyToClipboard() {{
                var copyText = document.getElementById("copyTarget");
                copyText.select();
                copyText.setSelectionRange(0, 99999); // for mobile

                navigator.clipboard.writeText(copyText.value).then(function() {{
                    alert("복사되었습니다!");
                }}).catch(function(err) {{
                    alert("복사 실패: " + err);
                }});
            }}
        </script>
    """, height=60)



def make_AgGrid(df):
    # JS code to <취소선> 표기 변경
    cell_renderer = JsCode("""
        function(params) {
            let val = params.value;

            // <s> 태그가 포함된 경우, 안쪽 텍스트 색상을 빨간색으로 설정
            if (val.includes("<s>")) {
                val = val.replace("<s>", "<취소선>")
                        .replace("</s>", "</취소선>");
            }
            return val;
        }
        """)
    

    # Grid 옵션 설정
    gb = GridOptionsBuilder.from_dataframe(df)
    # Enable pagination
    gb.configure_pagination(
    paginationAutoPageSize=False, 
    paginationPageSize=3
    )

    # Create tooltip JS code
    tooltip_renderer = JsCode("""
    class CustomTooltip {
        init(params) {
            this.eGui = document.createElement('div');
            this.eGui.innerHTML = '<div style="padding: 16px; background: #f5f5f5; border-radius: 8px;">' + 
                                '<strong>' + params.value + '</strong>' + '</div>';
        }
        getGui() {
            return this.eGui;
        }
    }
    """)

    # Enable tooltips for all columns
    for col in df.columns:
        gb.configure_column(
            col,
            tooltipField=col,  # Field used for tooltip
            valueFormatter=col    # Custom JS formatting for tooltip
        )


    # 첫 번째 컬럼에 체크박스 선택 기능 추가
    gb.configure_grid_options(domLayout='autoHeight')
    
    # Enable default tooltip
    gb.configure_grid_options(
        enableBrowserTooltips=True,  # 브라우저 기본 툴팁 사용
        tooltipShowDelay=0, 
        tooltipHideDelay=2000)
        
    # 툴팁 관련 전역 옵션
    gb.configure_grid_options(
        tooltipShowDelay=0,
        tooltipComponent='customTooltip',
        components={"customTooltip": tooltip_renderer}
    )

    # if agent_mode: 
    selection_mode, use_checkbox = "multiple", True
    gb.configure_selection(selection_mode=selection_mode, use_checkbox=use_checkbox, pre_selected_rows=[0])
    gb.configure_default_column(editable=True, wrapText=True, autoHeight=True)

    # 통화 단위 서식 지정
    currency_formatter = JsCode("""
        function(params) {
            if (params.value == null || params.value === undefined) {
                return '';
            }
            var decimalPoints = params.column.colDef.cellRendererParams.decimalPoints || 0;
            var currencySymbol = params.column.colDef.cellRendererParams.currencySymbol || '€';
            var value = params.value;

            // Format the number with thousand separators and decimal points
            var formattedNumber = value.toLocaleString('en-US', {
                minimumFractionDigits: decimalPoints,
                maximumFractionDigits: decimalPoints
            });

            return currencySymbol + formattedNumber;
        }
        """)

    currency_getter = JsCode("""
        function(params) {
            return params.data[params.colDef.field];
        }
        """)

    # AgGrid 테이블
    with st.expander("📃 :blue[**Results**]", expanded=True):
        # 칼럼 숨기기
        col100, col101 = st.columns([10, 1])
        with col100: pass
        with col101:
            popover = st.popover("✔️ Hide Columns", use_container_width=True)
            show_extra = popover.checkbox("EXTRA", True)
            show_credit = popover.checkbox("CREDIT", True)
            show_remark = popover.checkbox("REMARK", True)
        
        gb.configure_column("hull_no", filter="agSetColumnFilter")
        gb.configure_column("ship_type", filter="agSetColumnFilter")
        gb.configure_column("size", filter="agSetColumnFilter")
        gb.configure_column("extra", hide=not show_extra,
                            valueGetter=currency_getter,
                            valueFormatter=currency_formatter,
                            cellRendererParams={
                                'decimalPoints': 0,
                                'currencySymbol': '$',
                            }) 
        gb.configure_column("credit", hide=not show_credit,
                            valueGetter=currency_getter,
                            valueFormatter=currency_formatter,
                            cellRendererParams={
                                'decimalPoints': 0,
                                'currencySymbol': '$',
                            }) 
        gb.configure_column("remark", hide=not show_remark)
        gb.configure_column("buyer_comment", width=500, filter="agTextColumnFilter", cellRenderer=cell_renderer) 
        gb.configure_column("builder_reply", width=500, filter="agTextColumnFilter", cellRenderer=cell_renderer)

        grid_options = gb.build()
        grid_options["rowHeight"] = 60

        # 조건부 음영 효과 주기
        cellStyle = JsCode(
            r"""
            function(cellClassParams) {
                // 조건 1: extra > 0이면 금색 배경
                if (cellClassParams.colDef.field === 'extra' && cellClassParams.value > 0) {                    
                    return {'background-color': '#fad7d7'};
                }
                // 조건 1: credit > 0이면 금색 배경
                if (cellClassParams.colDef.field === 'credit' && cellClassParams.value > 0) {                    
                    return {'background-color': '#b6d5fa'};
                }

                // 조건 2: 셀 값에 <s> 태그가 포함되어 있으면 회색 배경
                if (cellClassParams.value && typeof cellClassParams.value === 'string' && cellClassParams.value.includes('<s>')) {
                    return {'background-color': '#e8fafa'};
                }

                return {};
            }
            """
        )
        grid_options['defaultColDef']['cellStyle'] = cellStyle

        grid_response = AgGrid(df, 
                            gridOptions=grid_options, 
                            editable=True, 
                            custom_css=custom_css, 
                            allow_unsafe_jscode=True,
                            enable_enterprise_modules=True,
                            fit_columns_on_grid_load=True,
                            theme="blue",   # options: streamlit, alpine, balham, material
                            )

    return grid_response



if __name__ == "__main__":
    st.title("Keyword Search")
    st.success("""
                워크 플로우
               - 키워드 서치 또는 하이브리드 서치
               - 검색 결과 표출 (AgGrid)
               - 검색 결과중 선택된 컨텍스트 베이스로 기술검토 생성
               """)
    
    with st.expander("🔎 :blue[**Search**]", expanded=True):
        st.success("검색구간")


    df = st.session_state.df.copy()

    grid_response = make_AgGrid(df=df) 
    new_df = grid_response['selected_rows']

    with st.expander("📂 :green[**Download**]", expanded=False):
            download_excel(df=new_df)

        
    with st.expander("🤖 :red[**Agent**]", expanded=True):
        try:
            new_df = grid_response['selected_rows']
            new_df = new_df.fillna("None")

            # 선택 내용을 질의 응답 페어로 묶어서 컨텍스트로 재세팅
            restr_context = ""
            for cmt, repl, extra, credit, remark in zip(new_df["buyer_comment"], new_df["builder_reply"], new_df["extra"], new_df["credit"], new_df["remark"]):
                temp = '<buyer_comment>' + "\n" + cmt + "\n" + "<builder_reply>" +"\n"+ repl + "\n"+ "<Extra> "+ "\n" + str(extra) + "\n"+ "<Credit> "+ "\n" + str(credit) +   "\n" + "<Remark> "+ "\n" + remark +"\n\n" + "-------------------------------------------------------------" + "\n\n"
                restr_context += temp

            col11, col22, col33, col44 = st.columns([5, 3, 5, 5])
            with col11: 
                txt1 = st.text_area(label=":green[**Selected Context**]", value=restr_context, key="wre", height=600)
                text_area_copy(txt1) 
            with col22:
                txt1 = txt1.replace("<s>", "~~").replace("</s>", "~~").replace("\\n", "<br>")
                st.markdown("취소선 체크")
                with st.container(height=600, border=True):
                    st.markdown(txt1)
            with col33: 
                txt2_1 = st.text_area(label=":green[**System Prompt**]", placeholder="설계 검토 시스템 프롬프트", value=input_template, key="dfsdf", height=250)
                txt2_2 = st.text_area(label=":green[**Buyer's New Comment**]", placeholder="검토대상 선주 요구사항 입력", value=example_input, key="dfdfsdf", height=300)
                text_area_copy(txt2_2) 
                
            with col44: 
                question = txt2_1 + "\n" + txt2_2   # 시스템 프롬프트 + 질문 쿼리
                context_input = txt1
                API_URL = "http://localhost:8000/invoke"

                if st.button("🔍 Agent Review", type="primary"):
                    if not question or not context_input:
                        st.warning("질문과 문맥을 모두 입력해주세요.")
                    else:
                        with st.spinner("Reviewing..."):
                            try:
                                # 입력 데이터 구성
                                payload = {
                                    "messages": [question],
                                    "context": context_input.strip().split("\n")
                                }

                                # FastAPI POST 요청
                                response = requests.post(API_URL, json=payload)
                                response.raise_for_status()

                                # 결과 출력
                                result = response.json()
                                txt3 = st.text_area(label=":green[**Agent Review**]", value=result['generation'], key="wsdfsdffer", height=530)
                                text_area_copy(txt3)       
                            except requests.exceptions.RequestException as e:
                                st.error(f"❌ 오류 발생: {e}")
            import base64
            col91, col92 = st.columns([4, 6])
            for path in new_df["remark"].tolist():
                if path.endswith("pdf"):
                    with open(path, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')  # 올바른 b64 인코딩
                    pdf_display = f"""
                        <iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>
                    """
                    with col91: st.markdown(pdf_display, unsafe_allow_html=True)
                elif path.endswith("jpg"):
                    with col92: st.image(path, caption="선택한 이미지", use_container_width=True)
                else:
                    pass

                
        except Exception as e:
            st.error("선택된 데이터가 없습니다.")
            pass
