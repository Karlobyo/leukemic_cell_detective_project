import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
    "leukemic_det/webinterface/model_dir/20240312-114546.h5")
