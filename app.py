import streamlit as st
#import cv2
import numpy as np
from PIL import Image

# OpenCV yüz algılama modeli yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dil seçenekleri
lang = st.sidebar.selectbox("Dil Seçin:", ["Türkçe", "English"])

# Başlık ve Açıklama
if lang == "Türkçe":
    st.title("Yüz Benzerlik Karşılaştırma Uygulaması")
    st.write("Fotoğraf yükleyin veya kameradan çekin.")
    first_image_label = "İlk fotoğrafı yükleyin:"
    second_image_label = "İkinci fotoğrafı yükleyin:"
    no_images_warning = "Her iki görsel kaynağı da yüklenmedi."
    similarity_label = "Benzerlik oranı:"
else:
    st.title("Face Similarity Comparison App")
    st.write("Upload a photo or take a picture.")
    first_image_label = "Upload the first photo:"
    second_image_label = "Upload the second photo:"
    no_images_warning = "Both visual sources have not been uploaded."
    similarity_label = "Similarity score:"

# Kullanıcı için seçenekler
option_1 = st.sidebar.selectbox("İlk görsel kaynağını seçin:", ["Fotoğraf Yükle", "Fotoğraf Çek (Kamera)"])
option_2 = st.sidebar.selectbox("İkinci görsel kaynağını seçin:", ["Fotoğraf Yükle", "Fotoğraf Çek (Kamera)"])

# Görüntüleri alacak değişkenler
image1, image2 = None, None

# İlk Görsel için Seçenekler
if option_1 == "Fotoğraf Yükle":
    uploaded_file_1 = st.file_uploader(first_image_label, type=["jpg", "jpeg", "png"], key="first_image")
    if uploaded_file_1:
        image1 = Image.open(uploaded_file_1)

elif option_1 == "Fotoğraf Çek (Kamera)":
    captured_image_1 = st.camera_input("Kameradan çekin:", key="camera1")
    if captured_image_1:
        image1 = Image.open(captured_image_1)

# İkinci Görsel için Seçenekler
if option_2 == "Fotoğraf Yükle":
    uploaded_file_2 = st.file_uploader(second_image_label, type=["jpg", "jpeg", "png"], key="second_image")
    if uploaded_file_2:
        image2 = Image.open(uploaded_file_2)

elif option_2 == "Fotoğraf Çek (Kamera)":
    captured_image_2 = st.camera_input("Kameradan çekin:", key="camera2")
    if captured_image_2:
        image2 = Image.open(captured_image_2)

# Benzerlik hesaplama ve gösterme
if image1 is not None and image2 is not None:
    # Yüz algılamak için her iki görüntüyü de işleyelim
    gray1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2GRAY)

    faces1 = face_cascade.detectMultiScale(gray1, 1.1, 4)
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)

    if len(faces1) > 0 and len(faces2) > 0:
        # İlk yüzü seç
        x1, y1, w1, h1 = faces1[0]
        x2, y2, w2, h2 = faces2[0]

        # Yüzleri yeniden boyutlandır
        face1_resized = cv2.resize(gray1[y1:y1+h1, x1:x1+w1], (100, 100))
        face2_resized = cv2.resize(gray2[y2:y2+h2, x2:x2+w2], (100, 100))

        # Benzerlik oranı hesapla
        similarity = cv2.matchTemplate(face1_resized, face2_resized, cv2.TM_CCOEFF_NORMED)[0][0] * 100

        # Sonuçları yan yana göster
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="İlk Fotoğraf", use_column_width=True)
        with col2:
            st.image(image2, caption="İkinci Fotoğraf", use_column_width=True)

        # Benzerlik oranını büyük şekilde göster
        st.markdown(f"<h1 style='text-align: center;'>{similarity_label} %{similarity:.2f}</h1>", unsafe_allow_html=True)
    else:
        st.warning(no_images_warning)
else:
    st.warning(no_images_warning)
