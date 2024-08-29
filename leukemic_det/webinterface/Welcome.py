import streamlit as st
from bg_loader import add_bg_from_local


st.set_page_config(
    page_title="Welcome",
    #page_icon="👋",
    layout='wide'
)


add_bg_from_local('leukemic_det/webinterface/images/lympho.png')


st.write("# Welcome to Leukemic Cell Detective!")

st.markdown("""### *Detecting healthy vs malignant cells*\n ### *from human white blood cells microscopic images*""")


#st.sidebar.success("Select a demo above.")

st.markdown('')

st.markdown('')

st.markdown('')


st.markdown(
"""
This is a research preview of a convolution neural network deep learning app\n
meant to deliver real-time predictions classifiying
human white blood cells microscopic images as healthy or malignant\n
(acute lymphoblastic leukaemia)
""")

st.markdown('')

st.markdown('')


st.markdown(
"""
Original dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223
""")

st.markdown('')

st.markdown("""
Dataset publication: Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019 [Data set]\n
The Cancer Imaging Archive https://doi.org/10.7937/tcia.2019.dc64i46r
""")
