#Descripion of what each explain component is 
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import math

import numpy as np
from wix_trino_client.trino_connection import WixTrinoConnection

tc = WixTrinoConnection()
predictions = []

class models:

    def __init__(self, name):
        self.name = name
        if self.name == '0 days':
            self.data_file = 'files/zero_days_x_train_fixed_3.csv'
            self.train_data = self.get_train_data()
            self.model_file = 'files/partner_detection_model_0_days_jan_24_final_3.pkl'
            self.model = self.get_model()
            self.production_table = 'prod.partners.partners_detection_zero_model_results'

        else:
            self.data_file = 'files/seven_days_x_train.csv'
            self.train_data = self.get_train_data()
            self.model_file = 'files/partner_detection_model_7_days_september_23_final.pkl'
            self.model = self.get_model()
            self.production_table = 'prod.partners.partners_detection_model_results'

        self.features_file = 'files/feature_discriptions.csv'
        self.feature_mapping = self.create_feature_mapping(self.model.feature_names_)
        self.feature_importance_dict = self.create_feature_importance_dict()
        self.unique_x_train_values = self.get_unique_values_as_dict()
        self.numeric_columns, self.categorical_columns, self.string_columns = self.data_types()
        self.predictions = []

    def get_model(self):
        with open(self.model_file, 'rb') as file:
            model = pickle.load(file)
            return model

    def get_train_data(self):
        x_train = pd.read_csv(self.data_file)
        # x_train.fillna('Null', inplace=True)
        return x_train

    def get_unique_values_as_dict(self):
        unique_values_dict = {}
        for column in self.train_data.columns:
            unique_values_dict[column] = self.train_data[column].unique().tolist()
        return unique_values_dict

    def data_types(self):
        numeric_columns = []
        string_columns = []
        categorical_columns = []

        for column in self.train_data.columns:
            # Check if the column has at least one int or float value
            if self.train_data[column].dtype in ['int64', 'float64']:
                numeric_columns.append(column)
                self.unique_x_train_values[column] = [int(x) if not math.isnan(x) else float('nan') for x in self.unique_x_train_values[column]]
            # Check if the column has at least one string value
            elif self.train_data[column].dtype == 'object':
                categorical_columns.append(column)
            else:
                string_columns.append(column)
        return numeric_columns, categorical_columns, string_columns

    def create_feature_importance_dict(self):
        feature_importance = self.model.get_feature_importance()
        feature_importance_dict = {feature_name: importance_score for feature_name, importance_score in zip(self.model.feature_names_, feature_importance)}
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        return sorted_features

    def create_feature_mapping(self, table):
        feature_names = table
        feature_mapping = {}
        for feature in feature_names:
            feature_mapping[feature] = feature.replace('_', ' ').title()

        return feature_mapping

    def create_avg_df(self):
        # Calculate mean for numeric columns
        new_data = self.train_data.iloc[0, :]
        for feature in self.train_data.columns:
            if feature in self.numeric_columns:
                new_data[feature] = self.train_data[feature].median().astype(int)
            else:
                new_data[feature] = self.train_data[feature].mode()
        new_data['email_match'] = None
        new_data = pd.DataFrame([new_data])
        return new_data

    # def modify_feature_values

    def playground_predict(self, special_features):
        X = self.create_avg_df()
        for feature in special_features.keys():
            X[feature] = special_features[feature]
        y_predict_proba = float(self.model.predict_proba(X)[:, 1][0])

        return y_predict_proba

    def investigate_predict(self, uuid):
        result = pd.DataFrame(tc.execute_sql_pandas(f'select * from {self.production_table} where uuid={uuid}'))
        prediction = result['is_web_pro'][0]
        prediction = round(prediction, 3)
        return prediction

    def plot_feature_importance(self, n):
        self.top_x_features_importance = self.feature_importance_dict[:n][::-1]
        self.top_x_features = [(self.feature_mapping[feature[0]], feature[1]) for feature in self.top_x_features_importance]
        self.top_x_features = pd.DataFrame(self.top_x_features).rename({0: 'Feature', 1: 'Importance'}, axis=1)

        # Create a horizontal bar chart with white bars and white axes
        fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the size as needed
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        ax.barh(self.top_x_features['Feature'], self.top_x_features['Importance'], color='white')

        # Customize the appearance of the chart
        ax.set_xlabel('Importance', fontsize=7, color='white')  # Change the x-axis label size
        ax.set_ylabel('Feature', fontsize=7, color='white')  # Change the y-axis label size
        ax.xaxis.set_tick_params(labelsize=7, colors='white')  # Change the x-axis tick label size and color
        ax.yaxis.set_tick_params(labelsize=7, colors='white')  # Change the y-axis tick label size and color

        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # Display the bar chart
        st.pyplot(fig)

        # self.feature_descriptions()

    def feature_descriptions(self, tab):
        self.feature_descriptions = pd.read_csv(self.features_file).rename({'feature':  'Feature', 'description': 'Description'},  axis=1)

        if tab == 'features':
            feature_mapping_table = pd.DataFrame(self.top_x_features_importance).rename({0: 'Feature', 1: 'Importance'}, axis=1)
            self.feature_descriptions = pd.merge(self.feature_descriptions,feature_mapping_table, left_on='Feature', right_on='Feature', how='inner')

            self.feature_descriptions['Feature'] = self.feature_descriptions['Feature'].map(self.feature_mapping)
            st.table(self.feature_descriptions[['Feature','Description','Importance']].sort_values(by='Importance', ascending=False).set_index('Feature'))

        elif tab == 'playground':
            combined_data = {}
            for index, row in self.feature_descriptions.iterrows():
                feature_name = row['Feature']
                if feature_name in self.feature_mapping:
                    fixed_feature_name = self.feature_mapping[feature_name]
                    description = row['Description']
                    combined_data[feature_name] = (fixed_feature_name, description)

            self.feature_mapping = combined_data







