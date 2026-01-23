import os
import shutil
from PIL import Image
import streamlit as st
from src.scripts.predict import *

UPLOAD_DIR = "static/uploads"
UPLOAD_FILE: str = ''
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title('IndFood Image Classification')
st.markdown('[source code](https://github.com/soumen7saha/IndFood-Image-Classification)')
# st.markdown('<a href="https://github.com/soumen7saha/IndFood-Image-Classification" target="_blank">source code</a>', unsafe_allow_html=True)

file = st.file_uploader('Upload Food Image', type=['jpg', 'jpeg', 'png'], max_upload_size=25)
file_location = None
if file:
    # print(file)
    UPLOAD_FILE = file.name
    file_location = os.path.join(UPLOAD_DIR, UPLOAD_FILE)
    with open(file_location, 'wb') as f:
        shutil.copyfileobj(file, f)
    
    # show the image
    image = Image.open(file_location).resize((200, 150))
    st.image(image, caption=UPLOAD_FILE)

model_rb = st.radio('Please select the CNN Model:', ['ResNet-152', 'ConvNext-S'])
model = None
if model_rb:
    if model_rb == 'ResNet-152':
        model = 'resnet'
    elif model_rb == 'ConvNext-S':
        model = 'convns'

btn = st.button('Classify Food')
if model_rb and file_location and btn:
    result = predict(FoodModel(img_url=file_location, model=model)).__dict__
    st.success(f'Uploaded food is ***{result["t1_class"]}***')
    st.write('\n**Top 5 Predicted Classes:**')
    c = 1
    for i in result['t5_preds']:
        st.write(f'{c}. *{i}*')
        c+=1
