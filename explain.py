#!/usr/bin/python
# -*- coding: utf-8 -*-




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

    pass





