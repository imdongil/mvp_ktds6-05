import streamlit as st

kt_qna_title = "KT TV 상담"

st.write(kt_qna_title)

# Define a list of options
options = ["KT_MAR4510C", "KT_KI1100", "KT_GiGA_Genie"]
with st.sidebar:
    st.write(kt_qna_title)
    selected_option = st.sidebar.selectbox("TV 모델 선택:", options)
    st.write(f"선택된 TV모델: {selected_option}")
    st.write("테스트 질문 예시")
    st.write("지니 TV의 인터넷 연결은 어떻게 하나요?")
    st.write("모바일 기가지니 앱은 어떻게 설치하나요?")
    st.write("PIN 번호는 무엇인가요?")

