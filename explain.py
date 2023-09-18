#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
import shap
import pickle
import os
import streamlit as st



def feature_importance_dict(model, fixed_feature_names):
    feature_importance = model.get_feature_importance()
    feature_importance_dict = {feature_name: importance_score for feature_name, importance_score in
                                   zip(fixed_feature_names, feature_importance)}
    # feature_importance_dict = {key.replace('_', ' ').title(): value for key, value in feature_importance_dict.items()}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    return sorted_features



def perm_import_plot():
    pass

def pdplot():
    pass


def shapValue(model, x_train):
    explainer = shap.Explainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(x_train)

    return shap_values

    shap.summary_plot(shap_values, x_train)




