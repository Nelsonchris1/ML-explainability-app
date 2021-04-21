import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots
import eli5
import pickle
import os



def perm_import(model, X_val, y_val, return_importances=False):
    # Load up model
    ml_model = pickle.load(open(model, "rb"))
    perm = PermutationImportance(ml_model, random_state=1).\
            fit(X_val, y_val)
    feat_name = X_val.columns.tolist()
    eli5_show_weights = eli5.show_weights(perm, 
                        feature_names=feat_name)
    
    importances = eli5.explain_weights_df(perm, feature_names=feat_name)
    
    if return_importances == True:
        return importances
    

def perm_import_plot(plot_importance):
    plt.figure(figsize=(10,8))

    plt.errorbar(x=importances['feature'],
                y = importances['weight'],
                yerr=importances['std'],
                capsize=8, fmt='none')
    plt.xticks(rotation = 90)
    sns.pointplot(x='feature',
                 y='weight',
                 data=importances,
                 dodge=True, join=False, ci='none')


#Partial dependeny plot

def pdplot(model, X_val, feat):
    ml_model = pickle.load(open(model, "rb"))
    feat_names = X_val.columns.tolist()
    pdp_assign = pdp.pdp_isolate(model = ml_model, dataset=X_val, model_features=feat_names, feature=feat)
    pdp.pdp_plot(pdp_assign, feat)
    plt.show()
    plt.savefig('tempdir/books_read.png')

def shapValue(model, x_train, x_val,tree_model, row_to_show=5):
    #open ml_model
    ml_model = pickle.load(open(model, "rb"))
    data_for_prediction = val_X.iloc[row_to_show]
    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    #when using tree model
    if tree_model:
        try:
            explainer = shap.TreeExplainer(ml_model)
            shap_values = explainer.shap_values(data_for_prediction)
            shap.initjs()
            return shap.force_plot(explainer.expected_value[1],
                                shap_values[1],
                                data_for_prediction)
        except Exception as e:
            print(e)

            
    else:
        explainer = shap.KernelExplainer(my_model.predict_proba, x_train)
        shap_values = explainer.shap_values(data_for_prediction)
        return shap.force_plot(explainer.expected_value[1],
                               k_shap_values[1],
                               data_for_prediction)
    
    