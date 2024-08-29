import streamlit as st
import base64
from leukemic_det.webinterface.bg_loader import add_bg_from_local

st.set_page_config(
    page_title="Intro",
    #page_icon="👋",
    layout='wide'
)


# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )


add_bg_from_local('/Users/carlobarbini/code/Karlobyo/leukemic_cell_det_project/leukemic_cell_detective_project/leukemic_det/webinterface/images/lympho.png')


st.write("# Welcome to Leukemic Cell Detective!")

st.markdown('### *Detecting healthy vs malignant cells from human white blood cells microscopic images*')


#st.sidebar.success("Select a demo above.")

st.markdown('')

st.markdown('')

st.markdown('')


st.markdown(
"""
This is a research preview of a convolution neural network deep learning app\n
meant to deliver real-time predictions classifiying
human white blood cells microscopic images as healthy or malignant
(acute lymphoblastic leukaemia)
""")

st.markdown('')

st.markdown(
"""
Original dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223

Dataset publication: Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]\n
The Cancer Imaging Archive https://doi.org/10.7937/tcia.2019.dc64i46r
""")
