# from fastapi import FastAPI
import streamlit as st
#import numpy as np
from PIL import Image
import base64
#import time
#import os
#import requests
#import wavfile
#import pretty_midi
#import io


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
def callback():
    st.session_state.button_clicked = True


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )



# app = FastAPI()

# # Define a root `/` endpoint
# @app.get('/')
# def index():
#     return {'ok': True}

# @app.get('/electronify')

st.set_page_config(layout='wide')

CSS = """
h1 {
    color: white;
}
h2 {color: black;
}
.stApp {
    background-image: webinterface/app_cloud.py;
    background-size: cover;
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.title('Leukemic Cell Detective')


st.markdown('### *Detecting healthy vs malignant cells from white blood cells microscopic images*')

#add_bg_from_local('images/bg_app.png')
