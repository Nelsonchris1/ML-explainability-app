#Descripion of what each explain component is 
import base64
import pickle
import shap
from explain import feature_importance_dict
from wix_trino_client.trino_connection import WixTrinoConnection
import pandas as pd



fixed_head = """
    <h1 style="font-size: 54px;text-align:center;">Web Pro Identification DS Model  </h1>
"""

Image = "ML-removebg.png"
shap_image = "SHAP.png"

opening_explanation = """
 <h1 style="font-size: 22px;font-weight: normal;text-align:center;">This app is aimed to help you understand how the web professional detection model is making 
it's predictions. Have Fun! </h1>
"""

feature_importance_title = """
    <h1 style="font-size: 36px;text-align:left;">Feature Importance </h1>
"""

with open('/Users/eyalk/partner_detection_model_7_days_september_23_final.pkl', 'rb') as file:
    model = pickle.load(file)

feature_names = model.feature_names_
fixed_feature_names = [name.replace('_', ' ').title() for name in feature_names]
feature_importance_dict = feature_importance_dict(model, fixed_feature_names)

data_table='sandbox.marketing.partners_detection_model_united_7_days_united'

# def train_val_test_split(data_table):
#     tc = WixTrinoConnection()
#     df = pd.DataFrame(tc.execute_sql_pandas(f'select * from {data_table}'))
#
#
#     x_train=df[df['purpose']=='Train'].drop(['uuid','purpose','label'],axis=1)
#     y_train=df[df['purpose']=='Train']['label']
#
#     x_val = df[df['purpose'] == 'Validation'].drop(['uuid', 'purpose', 'label'], axis=1)
#     y_val = df[df['purpose'] == 'Validation']['label']
#
#     x_test = df[df['purpose'] == 'Test'].drop(['uuid', 'purpose', 'label'], axis=1)
#     y_test = df[df['purpose'] == 'Test']['label']
#
#     return x_train, y_train, x_val, y_val, x_test, y_test, df

# x_train = pd.read_csv('seven_days_x_train.csv')
# print('got train df')
# Create an explainer object

# Calculate SHAP values
# shap_values = pd.read_parquet('seven_days_full_shap_values.parquet')
print('got shap values ')

