import os
from dotenv import load_dotenv

# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.core.exceptions import HttpResponseError, ClientAuthenticationError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from PIL import Image, ImageDraw, ImageFont  # pip install pillow
from io import BytesIO
import sys

load_dotenv()

# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
#     azure_deployment=os.getenv("AZURE_DEPLOYMENT_MODEL"),
# )

# embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
#     azure_deployment=os.getenv("AZURE_DEPLOYMENT_EMBED_NAME"),
# )

# 텍스트 파일 경로
# file_path = "KT_MAR4510C.txt"

# 파일 읽기
# with open(file_path, "r", encoding="utf-8") as file:
#    file_content = file.read()

# 개행문자 기준으로 나누기
##lines = file_content.split("\n")

##KT_MAR4510C_documents = lines

## KT_MAR4510C_vector_store = FAISS.from_texts(KT_MAR4510C_documents, embeddings)
## KT_MAR4510C_retrival = KT_MAR4510C_vector_store.as_retriever(search_kwargs={"K": 3})

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPEN_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZUER_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZUER_SEARCH_INDEX_NAME = os.getenv("AZUER_SEARCH_INDEX_NAME")
AZURE_DEPLOYMENT_MODEL = os.getenv("AZURE_DEPLOYMENT_MODEL")

try:
    search_credential = AzureKeyCredential(AZUER_SEARCH_KEY)

    # Initialize OpentAI client with API KEY
    openai_client = AzureOpenAI(
        api_version="2024-12-01-preview",
        api_key=AZURE_OPEN_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    # Initialize Search client with API KEY
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZUER_SEARCH_INDEX_NAME,
        credential=search_credential,
    )
except ClientAuthenticationError as e:
    print("Authentication error. Please check your API keys and endpoints.")
    sys.exit(1)
except HttpResponseError as e:
    print(f"HTTP response error: {e.message}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

# This prompt provides instructions to the model
ground_prompt = """
    You are a friendly assistant.
    Answer the query using only the sources provided below in a friendly and concise bulleted manner.
    Answer ONLY with the facts listed in the list of sources below.
    If there isn't enough information below, say you don't know.
    Do not generate answers that don't use the sources below.
    Query: {query}
    Sources: {sources}
    
    **출처**
    - (page source and page number) 
"""

# chat_prompt = ChatPromptTemplate.from_template(
#     """
#     Context 안에 있는 정보를 바탕으로 질문에 답해줘.

#     <context>
#     {query}
#     </context>

#     질문 : {sources}

#     **출처**
#     - (page source and page number)
#     """
# )

# Search results are created by the search client.
# Search results are composed of the top 5 results and the fields selected from the search index.
try:
    search_result = search_client.search(
        search_text=ground_prompt, top=5, select="title, chunk"
    )

    search_results_list = list(search_result)
    print(f"Found {len(search_results_list)} docements matching the query.")
except Exception as e:
    print(f"An unexpected error occurred during search: {e}")
    sys.exit(1)
# Search results include the top 5 matches to your query.

sources_formatted = "\n".join(
    [f'{document["title"]}:{document["chunk"]}' for document in search_results_list]
)


# 함수 정의
def create_rag_retreve(query):
    # print("==Search Results =")
    # print(sources_formatted)
    # Send the search results and the query to the LLM to generate a response based on the prompt.
    messages = [
        {
            "role": "user",
            "content": ground_prompt.format(query=query, sources=sources_formatted),
        }
    ]

    response = openai_client.chat.completions.create(
        model=AZURE_DEPLOYMENT_MODEL,
        messages=messages,
        temperature=0.3,
    )

    # print("== Response ==")
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


##document_chain = create_stuff_documents_chain(llm, prompt)
##retrival_chain = create_retrieval_chain(KT_MAR4510C_retrival, document_chain)

##프로젝트 타이틀 정의
kt_qna_title = "KT TV 상담"
st.set_page_config(page_title=kt_qna_title)
st.title(kt_qna_title)

st.page_link(
    "https://help.kt.com/servicetip/ServiceTipInfo.do",
    label="(참고)kt 간편한 셀프 해결 페이지 링크",
    icon="🌎",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "KT TV 상담을 해주는 도우미입니다."}
    ]

##To-Do List : 파일 업로드를 통한 모델 선택 구현 필요
uploaded_file = st.file_uploader(
    "상담 받고자 하는 TV 셋탑 사진을 등록 해 주세요!",
    type=["png", "jpg", "jpeg", "webp", "gif"],
)

if uploaded_file is not None:
    # 업로드된 파일을 Pillow Image 객체로 열기
    image = Image.open(uploaded_file)

    # 이미지 표시
    st.image(image, caption="업로드된 이미지", use_container_width=True)
    # 이미지 정보 출력
    # st.write("이미지 포맷:", image.format)
    # st.write("이미지 크기:", image.size)
    # st.write("이미지 모드:", image.mode)

    # 다시 바이너리로 변환
    img_bytes_io = BytesIO()
    image.save(img_bytes_io, format="BMP")  # API가 원하는 포맷으로 지정
    img_bytes = img_bytes_io.getvalue()
    image_data = img_bytes
    # with open(uploaded_file.name, "rb") as image_file:
    #     image_data = image_file.read()

    COMPUTER_VISION_KEY = os.getenv("COMPUTER_VISION_KEY")
    COMPUTER_VISION_ENDPOINT = os.getenv("COMPUTER_VISION_ENDPOINT")
    credential = AzureKeyCredential(COMPUTER_VISION_KEY)
    client = ImageAnalysisClient(
        endpoint=COMPUTER_VISION_ENDPOINT, credential=credential
    )

    result = client.analyze(
        image_data=image_data,
        # feature=["Tags", "Description", "Objects", "Brands"],
        visual_features=[
            VisualFeatures.TAGS,
            VisualFeatures.CAPTION,
            VisualFeatures.OBJECTS,
        ],
        model_version="latest",
    )

    # caption을 출력하는 부분
    if result.caption is not None:
        print(" Caption:")
        print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.2f}")
        st.write("Caption: " + result.caption.text)

    # 태그 출력하는 부분
    if result.tags is not None:
        print(" Tags:")
        tags = ""
        for tag in result.tags.list:
            print(f" - '{tag.name}', Confidence {tag.confidence:.2f}")
            tags += tag.name + ", "
        st.write("Tags: " + tags)

# 채팅 처리 부분
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

        # content 필드만 추출
        contents = [message["content"] for message in st.session_state.messages]
        # 출력
        # for idx, content in enumerate(contents, start=1):
        #     print(f"Message {idx}: {content}")
        # print()
        last_question = contents[-1]
        # print(query)

        # result = retrival_chain.invoke({"input": query})
        # response_text += result["answer"]
        result = create_rag_retreve(last_question)
        response_text += result
        placeholder.markdown(response_text)
    st.session_state["messages"].append({"role": "assistant", "content": response_text})

# 셋탑 선택 상자 : 기본은 채팅에서 선택되겠으나 테스트를 위해서 별도 표기
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
