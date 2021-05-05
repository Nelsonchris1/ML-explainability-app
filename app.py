import streamlit as st
from explain import pdplot, perm_import, perm_import_plot, shapValue
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import json
from io import StringIO


html_txt = """<font color='blue'>Upload files to Explain</font>"""

st.write("Explain My")

option = st.sidebar.selectbox("Select view", ("Home", "ML Explain", "Tutorial"))

if option == "ML Explain":
    st.sidebar.write(html_txt, unsafe_allow_html=True)
    train = st.sidebar.file_uploader("X_train", type=["csv", "text"])
    if train :
        X_train = pd.read_csv(train)

    st.write("""
        #
        """)

    test = st.sidebar.file_uploader("X_test", type=["csv", "text"])
    if test is not None:
        X_test = pd.read_csv(test)
        
    st.write("""
        #
        """)

    train_y = st.sidebar.file_uploader("y_train", type=["csv", "text"])
    if train_y is not None:
        y_train = pd.read_csv(train_y)
        
    st.write("""
        #
        """)

    test_y = st.sidebar.file_uploader("y_test", type=["csv", "text"])
    if test_y is not None:
        y_test = pd.read_csv(test_y)
    st.write("""
        #
        """)

    model = st.sidebar.file_uploader("model")
    st.write("""
        #
        """)

    if model is not None:
        with open(os.path.join('tempdir', 'model2'),"wb") as f: 
            f.write(model.getbuffer())

    features = st.sidebar.file_uploader("Upload feature as txt")
    st.write("""
    #
    """)

    

    # if  st.sidebar.button("Submit upload files"):
    
    radio_option = ["None","Permutation Importance","Partial Density Plot", "Shap Values", "All"]
    selected_explain = st.radio("Choose page:", radio_option)

    if selected_explain == "Permutation Importance":
         #firstly compute importance and then plot perm_importance _plot
        importances = perm_import(model='tempdir/model2', X_val=X_test, y_val=y_test, return_importances=True)
        st.dataframe(importances)
        perm_import_plot(importance=importances)
        
    elif selected_explain == "Partial Density Plot":
        if features:
            stringio = StringIO(features.getvalue().decode('utf-8'))
            feat_col = [feature.strip() for feature in stringio.readlines()]
            feat_col_name = feat_col
            feat_selected = st.selectbox("select base column name", feat_col_name)
            comapre_select = st.radio("Comapre plot", ["No", "Yes"])
            if comapre_select == "Yes":
                feat_compare_selected = st.selectbox("select coulumn to comapre", feat_col_name)
                pdplot("tempdir/model2", X_test, feat_selected)
                pdplot("tempdir/model2", X_test, feat_compare_selected, "img_pdplot2.png")
                st.image('tempdir/img_pdplot.png')
                st.image('tempdir/img_pdplot2.png')
            else:
                pdplot("tempdir/model2", X_test, feat_selected)
                st.image('tempdir/img_pdplot.png')

    elif selected_explain == "Shap Values":
        feat_select_shap = st.selectbox("select num of rows to predict", [0, 5, 10, 20, 30, 40, 50, 100, 200, 300])
        if feat_select_shap != 0:
            shapValue("tempdir/model2", X_train, X_test, tree_model=False, row_to_show=feat_select_shap)
            plt.savefig("tempdir/shapvalue.png",dpi=500, bbox_inches='tight')
            st.image('tempdir/shapvalue.png')

    elif selected_explain == "All":
        importances = perm_import(model='tempdir/model2', X_val=X_test, y_val=y_test, return_importances=True)
        st.dataframe(importances)
        perm_import_plot(importance=importances)

        if features:
            stringio = StringIO(features.getvalue().decode('utf-8'))
            feat_col = [feature.strip() for feature in stringio.readlines()]
            feat_col_name = feat_col
            feat_selected = st.selectbox("select base column name", feat_col_name)
            pdplot("tempdir/model2", X_test, feat_selected)
            st.image('tempdir/img_pdplot.png')

        #dispaly shap values
        shapValue("tempdir/model2", X_train, X_test, tree_model=False, row_to_show=feat_select_shap)
        plt.savefig("tempdir/shapvalue.png",dpi=500, bbox_inches='tight')
        st.image('tempdir/shapvalue.png')
    

    else:
        st.write("Click on any ML_explain to explain Model")
            