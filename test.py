import lime
import lime.lime_tabular
import pickle
import pandas as pd
import streamlit as st
import numpy as np
from io import StringIO

def lime_explain(x_train, x_val, y_train, feat, model, i):
    ml_model = pickle.load(open(model, 'rb'))
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                                        feature_names = feat, 
                                                        class_names = ['True', 'False'], 
                                                        mode='classification', 
                                                        training_labels=x_train.columns.values.tolist())
            
    predict_fn = lambda x: ml_model.predict_proba(x).astype(float)
    exp = explainer.explain_instance(x_val.values[i], predict_fn, num_features = 5)
    exp.save_to_file('lime.html')


X_train = pd.read_csv('train_X.csv')
X_test = pd.read_csv("test_X.csv")
y_train = pd.read_csv('y_test.csv')
with open('feat_text.txt') as f:
    contents = f.readlines()

uploaded_file =  st.file_uploader('', type='txt', accept_multiple_files=False)
if uploaded_file:
    stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
    features = [feature.strip() for feature in stringio.readlines()]



Click_lime = st.button("Run lime")
if st.button:
    lime_explain(x_train=X_train.astype('float'), x_val=X_test.astype('float'),
                                     y_train = y_train.astype('float'),
                                     feat=features, model='model', i=0)

    st.write("Done Running")



