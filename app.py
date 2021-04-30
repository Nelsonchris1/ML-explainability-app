import streamlit as st
from explain import pdplot, perm_import, perm_import_plot
import pandas as pd
import os
import pickle
import numpy as np
import json
from io import StringIO



st.write("""
    # My First App
    #
""")

train = st.file_uploader("X_train")
if train is not None:
    X_train = pd.read_csv(train)

st.write("""
    #
     """)

test = st.file_uploader("X_test", type=["csv", "text"])
if test is not None:
    X_test = pd.read_csv(test)
    
st.write("""
    #
     """)

train_y = st.file_uploader("y_train")
if train_y is not None:
    y_train = pd.read_csv(train_y)
    
st.write("""
    #
     """)

test_y = st.file_uploader("y_test")
if test_y is not None:
    y_test = pd.read_csv(test_y)
st.write("""
    #
     """)

model = st.file_uploader("model")
st.write("""
    #
     """)

features = st.file_uploader("Upload feature as txt")
st.write("""
    #
""")


if features:
    stringio = StringIO(features.getvalue().decode('utf-8'))
    feat_col = [feature.strip() for feature in stringio.readlines()]
    feat_col_name = feat_col
    st.selectbox("select column name", feat_col_name)

        
# def write_read_txt():
#     with open(os.path.join('temp_feat', 'text.txt'), 'w') as f:
#         f.write(json.dumps(str(features)))

#     with open(os.path.join('temp_feat', 'text.txt'), "r") as f:
#         df_features = json.loads(f.read())
#     return df_features

# df_feat = write_read_txt()

if model is not None:
    with open(os.path.join('tempdir', 'model2'),"wb") as f: 
        f.write(model.getbuffer())
# st.write(df_feat)
# optipon = st.selectbox('',
#             tuple(df_feat))
# st.write('You selected:', optipon)


allFiles = [X_train, X_test, y_train, y_test, model]





if  st.button("Submit upload files"):
    # cleared = True
    # for eachFile in allFiles:
    #     if eachFile != True:
    #         st.warning("All files need to e uploaded before submitting")
    #         # cleared = False
    #         break

    # pdplot("tempdir/model2", test_data, "Ball Possession %")

    #firstly compute importance and then plot perm_importance _plot
    importances = perm_import(model='tempdir/model2', X_val=X_test, y_val=y_test, return_importances=True)
    st.dataframe(importances)
    # st.dataframe(importances)
    perm_import_plot(importance=importances)

    # st.image("tempdir/img_pdplot.png")
        
    # if cleared: 
       
        # pdplot(model, X_val, feat)
    