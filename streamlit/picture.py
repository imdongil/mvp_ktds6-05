import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from PIL import Image, ImageDraw, ImageFont  # pip install pillow
from io import BytesIO

import streamlit as st

load_dotenv()

COMPUTER_VISION_KEY = os.getenv("COMPUTER_VISION_KEY")
COMPUTER_VISION_ENDPOINT = os.getenv("COMPUTER_VISION_ENDPOINT")

credential = AzureKeyCredential(COMPUTER_VISION_KEY)

client = ImageAnalysisClient(endpoint=COMPUTER_VISION_ENDPOINT, credential=credential)


def get_image_info():
    file_path = input("Enter image file path: ")

    with open(file_path, "rb") as image_file:
        image_data = image_file.read()

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

    # 태그 출력
    if result.tags is not None:
        print(" Tags:")
        for tag in result.tags.list:
            print(f" - '{tag.name}', Confidence {tag.confidence:.2f}")

    # 이미지 그리기 준비
    image = Image.open(file_path)
    draw = ImageDraw.Draw(image)

    # object 출력
    if result.objects is not None:
        print(" Objects:")
        for object in result.objects.list:
            print(
                f" - '{object.tags[0].name}', Confidence: {object.tags[0].confidence:.2f} at location {object.bounding_box}"
            )
            x, y, w, h = (
                object.bounding_box["x"],
                object.bounding_box["y"],
                object.bounding_box["w"],
                object.bounding_box["h"],
            )
            print(x, y, w, h)
            draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=2)
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((x + 2, y + 2), object.tags[0].name, fill="blue", font=font)


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
        st.write(" Caption:" + result.caption.text)

    # 태그 출력
    if result.tags is not None:
        print(" Tags:")
        for tag in result.tags.list:
            print(f" - '{tag.name}', Confidence {tag.confidence:.2f}")

    # 이미지 그리기 준비
    image = image_data  # Image.open(file_path)
    draw = ImageDraw.Draw(image)

    # object 출력
    if result.objects is not None:
        print(" Objects:")
        for object in result.objects.list:
            print(
                f" - '{object.tags[0].name}', Confidence: {object.tags[0].confidence:.2f} at location {object.bounding_box}"
            )
            x, y, w, h = (
                object.bounding_box["x"],
                object.bounding_box["y"],
                object.bounding_box["w"],
                object.bounding_box["h"],
            )
            print(x, y, w, h)
            draw.rectangle(((x, y), (x + w, y + h)), outline="red", width=2)
            font = ImageFont.truetype("arial.ttf", 20)
            draw.text((x + 2, y + 2), object.tags[0].name, fill="blue", font=font)

    # st.image(image, caption="Uploaded Image.", use_column_width=True)
    # st.write("")
    # st.write("Classifying...")
    # get_image_info()
