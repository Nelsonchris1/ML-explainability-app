import streamlit as st

model = st.sidebar.file_uploader('model')

for files in model:
    if model is not None:
        with open(files) as f:
            st.sidebar.write(files)


    