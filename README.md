# mvp_ktds6-05
MS AI 개발역량 향상과정 MVP 제출

프로젝트 : 간편한 셀프 해결

내용 : kt 상품 가입 고객이 문제가 생겼을때 간편 해결 방법을 안내 하고 연관 상품을 안내하는 AI을 개발해 보고자 합니다.

플로우
1. 고객이 해결 하고자 하는 정보 대해서 질의 : 텍스트 또는 상품 캡춰 사진 업로드로 판단
2. 텍스트 또는 사진 기반으로 고객의 상품이나 해결 해야 하는 부분 탐지
3. 현재는Computer Vision을 통한 태그 및 캡션 인식까지만 구현함
4. 특정 상품에 대해서 해결 안내 가능한 내용 사전 학습 : 텍스트 또는 벡터 기반
5. Storage account 이용 하여 연관 상품 가이드 PDF 업로드
6. Azure AI Search 이용 업로든 된 정보를 인덱싱
7. 현재 구현은 json 파일을 BLOB에 업로드 하여 벡터 기반으로 인덱스 생성하여 안내 하도록 함
8. 학습된 내용 기반으로 안내를 하는데 안내 템플릿을 사전 정의 : 정의 내용에 연관 상품 추천 포함
9. ChatPromptTemplate을 이용하여 템플릿 형태로 안내하도로 개발
10. 연관 상품 추천하는 부분은 고민중
11. 템플릿 기반으로 LLM 안내
12. AzureChatOpenAI 기반으로 gpt-4.1-mini 모델기반으로 개발
13. 웹사이트에 업로드
14. streamlit 기반으로 웹환경 개발
15. Azure Web App생성하여 퍼블리싱
16. 최종 업로드 사이트 링크
17. https://ktds6-05-webapp-0929-dmdqcrfzbfbhf4cm.koreacentral-01.azurewebsites.net/
    
</br>
<P>
참고사항</br>
1. Storage account 이용</br>
	BLOB 스토리지 생성</br>
	Create a storage account</br>
		Storage account name : ktds605storage0929</br>
		Preferred storage type : Blob Storage</br>
		Redundancy : LRS</br>
		New container : pdf, json</br>
		Upload blob : KT_KI1100.pdf, kttv.json</br>
	Blob anonymous access</br>
		Change access level : Change the access level of container 'pdf'. => container</br>
	URL : </br>
		https://ktds605storage0929.blob.core.windows.net/pdf/KT_MAR4510C.pdf</br>
		https://ktds605storage0929.blob.core.windows.net/json/kttv.json</br>
</br>
2. Azure OpenAI 이용</br>
	Create a new Azure OpenAI service</br>
		Name : ktds6-05-openai-0929</br>
</br>
3. embedding model 생성</br>
	No deployments available with an embedding model.</br>
	배포 이름: dev-gpt-4.1-mini</br>
	배포 이름: dev-text-embedding-3-large</br>
</br>	
4. RAG : Azure AI Search 이용</br>
	Create a search service</br>
		Service name : ktds6-05-aisearch-0929</br>
		Select Pricing Tier : Basic</br>
		Parsing mode : Default</br>	
	import(New)</br>
		Model deployment : dev-text-embedding-3-large</br>	
	Data source name : ktds6-05-pdfdata-0929</br>
</br>	
5.Create Web App</br>
	Name : ktds6-05-webapp-0929</br>
	Linux Plan (Korea Central) : ktds6-05-plan-0929</br>
	URL : </br>
	https://ktds6-05-webapp-0929-dmdqcrfzbfbhf4cm.koreacentral-01.azurewebsites.net/</br>
</br>
6.Create Computer Vision</br>
	name : ktds6-05-computervision-093001</br>
</P>
