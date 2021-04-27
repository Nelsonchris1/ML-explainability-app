import streamlit as st
from io import StringIO


uploaded_file = st.file_uploader("Add text file !")
if uploaded_file:
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    features = [feature.strip() for feature in stringio.readlines()]
    
    st.write(features[0])
