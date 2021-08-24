#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots
import eli5
import shap
import pickle
import lime
import lime.lime_tabular
import os
import streamlit as st


def perm_import(
    model,
    X_val,
    y_val,
    score,
    return_importances=False,
    ):

    # Load up model

    ml_model = pickle.load(open(model, 'rb'))
    perm = PermutationImportance(ml_model, scoring=score,
                                 random_state=1).fit(X_val, y_val)
    feat_name = X_val.columns.tolist()
    eli5_show_weights = eli5.show_weights(perm, feature_names=feat_name)

    importances = eli5.explain_weights_df(perm, feature_names=feat_name)

    if return_importances == True:
        return importances


def perm_import_plot(importance):
    fig = plt.figure(figsize=(10, 8))

    plt.errorbar(x=importance['feature'], y=importance['weight'],
                 yerr=importance['std'], capsize=8, fmt='none')
    plt.xticks(rotation=90)
    sns.pointplot(
        x='feature',
        y='weight',
        data=importance,
        dodge=True,
        join=False,
        ci='none',
        )
    st.pyplot(fig)


# Partial dependeny plot

def pdplot(
    model,
    X_val,
    feat,
    image_name='img_pdplot.png',
    ):
    ml_model = pickle.load(open(model, 'rb'))
    feat_names = X_val.columns.tolist()
    pdp_assign = pdp.pdp_isolate(model=ml_model, dataset=X_val,
                                 model_features=feat_names,
                                 feature=feat)
    pdp.pdp_plot(pdp_assign, feat)
    plt.show()
    plt.savefig(image_name)


def shapValue(
    model,
    x_train,
    x_val,
    tree_model,
    row_to_show=5,
    ):

    # open ml_model

    ml_model = pickle.load(open(model, 'rb'))
    X_train_summary = shap.kmeans(x_train, 50)
    data_for_prediction = x_val.iloc[row_to_show]
    data_for_prediction_array = data_for_prediction.values.reshape(1,
            -1)

    # when using tree model

    if tree_model:
        try:
            explainer = shap.TreeExplainer(ml_model)
            shap_values = explainer.shap_values(data_for_prediction)
            shap.initjs()
            return shap.force_plot(explainer.expected_value[1],
                                   shap_values[1], data_for_prediction,
                                   matplotlib=True, show=False)
        except Exception as e:
            print (e)
    else:

        explainer = shap.KernelExplainer(ml_model.predict_proba,
                X_train_summary)
        shap_values = explainer.shap_values(data_for_prediction)
        return shap.force_plot(explainer.expected_value[1],
                               shap_values[1], data_for_prediction,
                               matplotlib=True, show=False)



def lime_explain(x_train, x_val, feat, model, i, targets, target_name):
    explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, 
                                                        feature_names = feat, 
                                                        class_names = [targets], 
                                                        mode='classification', 
                                                        training_labels=[target_name])
            
    predict_fn = lambda x: model.predict_proba(x).astype(float)
    exp = explainer.explain_instance(x_val.values[i], predict_fn, num_features = 5)
    exp.save_to_file('lime_explainer.html')
