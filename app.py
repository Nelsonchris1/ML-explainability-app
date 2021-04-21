import streamlit as st
from explain import pdplot
import pandas as pd
import os
import pickle
import numpy as np



st.write("""
    # My First App
    #
""")

train_X = st.file_uploader("X_train")

st.write("""
    #
     """)

test = st.file_uploader("X_test", type=["csv", "text"])
if test is not None:
    test_data = pd.read_csv(test)
    
st.write("""
    #
     """)

y_train = st.file_uploader("y_train")
st.write("""
    #
     """)

y_test = st.file_uploader("y_test")
st.write("""
    #
     """)

model = st.file_uploader("model")
st.write("""
    #
     """)
if model is not None:
    with open(os.path.join('tempdir', 'model2'),"wb") as f: 
        f.write(model.getbuffer())

allFiles = [train_X, test_data, y_train, y_test, model]




if  st.button("Submit upload files"):    
    # cleared = True
    # for eachFile in allFiles:
    #     if eachFile != True:
    #         st.warning("All files need to e uploaded before submitting")
    #         # cleared = False
    #         break

    pdplot("tempdir/model2", test_data, "Corners")
        
    # if cleared:
       
        # pdplot(model, X_val, feat)
    