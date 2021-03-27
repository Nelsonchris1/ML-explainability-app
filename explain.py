import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eli5 import PermutationImportance
import eli5
import pickle

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
    feat_names = X_val.columns.tolist()
    pdp_assign = pdp.pdp_isolate(model = model, dataset=X_val, model_features=feat_names, feature=feat)
    pdp.pdp_plot(pdp_assign, feat)
    plt.show()