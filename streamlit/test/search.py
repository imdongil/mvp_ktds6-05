import os

# AZURESEARCH_FIELDS_CONTENT: 자신의 인덱스에서 검색 대상을 의미하는 필드명
# AZURESEARCH_FIELDS_CONTENT_VECTOR: 'AZURESEARCH_FIELDS_CONTENT' 의 임베딩을 의미하는 필드명
# azure portal 에서 인덱싱 하셨다면, 기본으로 chunk, text_vector 로 되어있을 것입니다.
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "chunk"
os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "text_vector"

# 주의! AzureSearch 를 import 하기 전에 위 환경변수 셋팅을 완료해야한다.
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

AZURE_DEPLOYMENT_EMBED_NAME = os.getenv("AZURE_DEPLOYMENT_EMBED_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")

azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_API_KEY")
search_index_name = os.getenv("AZUER_SEARCH_INDEX_NAME")

# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=AZURE_DEPLOYMENT_EMBED_NAME,
#     openai_api_version="2024-12-01-preview",
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_key=AZURE_OPENAI_KEY,
# )
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    azure_deployment="dev-text-embedding-3-large",
)


vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=azure_search_endpoint,  # ai search 서비스의 엔드포인트
    azure_search_key=AZURE_SEARCH_KEY,  # ai search 서비스의 키
    index_name=search_index_name,
    search_type="hybrid",  # hybrid 가 기본 값이다. 가능한 값: 'similarity', 'similarity_score_threshold', 'hybrid', 'hybrid_score_threshold', 'semantic_hybrid', 'semantic_hybrid_score_threshold'
    embedding_function=embeddings.embed_query,
    additional_search_client_options={
        "retry_total": 3,
        "api_version": "2025-05-01-preview",
    },
)


@tool
def search_ktds(query, top_k=3):
    """
    ktds 회사의 비즈니스 정보를 검색합니다.

    Args:
        query (str): 검색할 키워드 또는 질문입니다. 예시: 'ktds의 빅데이터 사업'
        top_k (int, optional): 반환할 결과의 개수입니다. 기본값은 3입니다. 예시: 5

    Returns:
        list: 검색 결과 리스트
    """
    retriever = vector_store.as_retriever(k=top_k)

    results = [d for d in retriever.invoke(query)]

    return [doc for doc in results]
