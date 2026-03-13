# ----------------------------------
# 라이브러리
# ----------------------------------
import streamlit as st
import os
import zipfile
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from pathlib import Path

# ----------------------------------
# 제목
# ----------------------------------
st.title("이미지 자동 군집 프로그램")

st.write("이미지를 업로드하면 자동으로 군집화 후 ZIP으로 다운로드합니다.")


if st.button("새로 시작"):

    shutil.rmtree("temp_images", ignore_errors=True)

    st.session_state.upload_key += 1   # uploader 초기화

    st.rerun()

# ----------------------------------
# 업로드
# ----------------------------------
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

uploaded_files = st.file_uploader(
    "이미지 업로드",
    accept_multiple_files=True,
    type=["jpg","jpeg","png","bmp"],
    key=f"uploader_{st.session_state.upload_key}"
)

n_clusters = st.slider("군집 개수를 선택하세요",2,10,5)

# ----------------------------------
# 실행
# ----------------------------------
if st.button("군집 실행"):

    if not uploaded_files:
        st.warning("이미지를 업로드하세요")
        st.stop()

    # -----------------------------
    # 임시 폴더 생성
    # -----------------------------
    base_dir = "temp_images"
    input_dir = os.path.join(base_dir,"input")
    result_dir = os.path.join(base_dir,"result")

    shutil.rmtree(base_dir,ignore_errors=True)

    os.makedirs(input_dir)
    os.makedirs(result_dir)

    # -----------------------------
    # 업로드 이미지 저장
    # -----------------------------
    st.write("이미지 저장중...")

    image_paths = []

    for file in uploaded_files:

        path = os.path.join(input_dir,file.name)

        with open(path,"wb") as f:
            f.write(file.getbuffer())

        image_paths.append(path)

    st.success(f"{len(image_paths)}개 이미지 저장 완료")

    # -----------------------------
    # 이미지 전처리
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # -----------------------------
    # CNN 모델
    # -----------------------------
    with st.spinner("CNN 모델 로딩중..."):

        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()

    # -----------------------------
    # 특징 추출
    # -----------------------------
    st.write("이미지 특징 추출중...")

    features = []

    progress = st.progress(0)

    for i,path in enumerate(image_paths):

        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            feature = model(img)

        feature = feature.squeeze().numpy()

        features.append(feature)

        progress.progress((i+1)/len(image_paths))

    features = np.array(features)

    # -----------------------------
    # 군집
    # -----------------------------
    st.write("이미지 군집 실행중...")

    kmeans = KMeans(n_clusters=n_clusters,random_state=0)

    labels = kmeans.fit_predict(features)

    # -----------------------------
    # 결과 폴더 생성
    # -----------------------------
    cluster_dirs = []

    for i in range(n_clusters):

        d = os.path.join(result_dir,f"cluster_{i}")
        os.makedirs(d)

        cluster_dirs.append(d)

    # -----------------------------
    # 이미지 정리
    # -----------------------------
    log_placeholder = st.empty()
    log_messages = []
    
    import time

    for i,(path,label) in enumerate(zip(image_paths,labels)):
    
        filename = os.path.basename(path)
        dst = os.path.join(cluster_dirs[label],filename)
        shutil.copy(path,dst)
    
        # 로그 추가
        log_messages.append(f"{filename} → cluster-{label} 로 군집")
    
        # HTML 로그창 출력
        log_html = "<br>".join(log_messages)
    
        log_placeholder.markdown(
            f"""
            <div style="
                height:300px;
                overflow-y:scroll;
                background:black;
                color:lime;
                padding:10px;
                font-family:monospace;
                border-radius:8px;">
            {log_html}
            </div>
            """,
            unsafe_allow_html=True
        )
    
        progress.progress((i+1)/len(image_paths))
    
        time.sleep(0.05)
    


    st.success("군집 완료")

    # -----------------------------
    # ZIP 생성
    # -----------------------------
    zip_path = "cluster_result.zip"

    with zipfile.ZipFile(zip_path,"w") as zipf:

        for root,dirs,files in os.walk(result_dir):

            for file in files:

                full_path = os.path.join(root,file)

                arcname = os.path.relpath(full_path,result_dir)

                zipf.write(full_path,arcname)

    # -----------------------------
    # 다운로드
    # -----------------------------
    with open(zip_path,"rb") as f:

        st.download_button(
            label="결과 ZIP 다운로드",
            data=f,
            file_name="cluster_result.zip",
            mime="application/zip"
        )