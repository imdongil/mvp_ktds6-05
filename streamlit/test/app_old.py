import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain.vectorstores import FAISS

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_MODEL"),
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_EMBED_NAME"),
)

# 텍스트 파일 경로
file_path = "KT_MAR4510C.txt"

# 파일 읽기
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()

# 개행문자 기준으로 나누기
lines = file_content.split("\n")

KT_MAR4510C_documents = lines

KT_MAR4510C_vector_store = FAISS.from_texts(KT_MAR4510C_documents, embeddings)
KT_MAR4510C_retrival = KT_MAR4510C_vector_store.as_retriever(search_kwargs={"K": 3})

prompt = ChatPromptTemplate.from_template(
    """
    Context 안에 있는 정보를 바탕으로 질문에 답해줘.

    <context>
    {context}
    </context>

    질문 : {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrival_chain = create_retrieval_chain(KT_MAR4510C_retrival, document_chain)

kt_qna_title = "KT TV 상담"

# st.set_page_config(
#     page_title="KTDS 채팅 상담",
#     page_icon="🧊",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         "Get Help": "https://www.extremelycoolapp.com/help",
#         "Report a bug": "https://www.extremelycoolapp.com/bug",
#         "About": "# This is a header. This is an *extremely* cool app!",
#     },
# )

st.title(kt_qna_title)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "KT TV 상담을 해주는 도우미입니다."}
    ]

uploaded_file = st.file_uploader(
    "상담 받고자 하는 TV 셋탑 사진을 업로두 해 주세요!", type="png"
)

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if prompt := st.chat_input("User: "):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()

        # for chunk in llm.stream(st.session_state["messages"]):
        #     response_text += chunk.content
        #     placeholder.markdown(response_text)

        # for message in st.session_state.messages:
        #     with st.chat_message(message["role"]):
        #         query = message["content"]
        # content 필드만 추출
        contents = [message["content"] for message in st.session_state.messages]
        # 출력
        # for idx, content in enumerate(contents, start=1):
        #     print(f"Message {idx}: {content}")
        # print()
        query = contents[-1]
        # print(query)

        result = retrival_chain.invoke({"input": query})
        response_text += result["answer"]
        placeholder.markdown(response_text)
    st.session_state["messages"].append({"role": "assistant", "content": response_text})

# Define a list of options
options = ["KT_KI1100", "KT_MAR4510C"]
with st.sidebar:
    st.write(kt_qna_title)
    selected_option = st.sidebar.selectbox("상담받고 싶은 TV 모델 선택:", options)
    st.write(f"선택된 TV모델: {selected_option}")
    st.write("--------------------")
    st.write("테스트 질문 예시 :KT_MAR4510C")
    st.write("지니 TV의 인터넷 연결은 어떻게 하나요?")
    st.write("--------------------")
    st.write("테스트 질문 예시 :KT_KI1100")
    st.write("모바일 기가지니 앱은 어떻게 설치하나요?")
