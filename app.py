#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components

#explain is a custom file available in this repo
from explain import pdplot, perm_import, perm_import_plot, shapValue, lime_explain

#desc is another custom-made file available in this repo
from desc import descriptive_message_temp as desc
from desc import code, code2, overview_desc, home_page, fixed_head, code3, about_me

from remove import remove_files
import pandas as pd

#matlplot lib used to do graph functions
import matplotlib.pyplot as plt

import os
import stat
import random 
import numpy as np
from io import StringIO


def read_txt_and_pdplot(feature, test_X, ML_model):
                    feat_selected = st.selectbox('select base column name',
                            feature)
                    pdplot(ML_model, test_X, feat_selected)
                    st.image('img_pdplot.png')


def plot_perm_importance(test_x, test_y, score, ML_model):
                importances = perm_import(model=ML_model,
                                        X_val=test_x, y_val=test_y,
                                        score=score, return_importances=True)
                st.dataframe(importances)
                perm_import_plot(importance=importances)


def plot_shap_values(len_x, train_x, test_x, ML_model):

                random_selector =  st.button('Random_row', key="shap_002")

                if random_selector:
                    random_num = random.randint(0, len_x)
                    st.write(f"Displaying for row number {random_num}")
                    shapValue(ML_model, train_x, test_x, 
                            tree_model=False, row_to_show=random_num)
                    plt.savefig('shapvalue.png', dpi=500,
                                bbox_inches='tight')
                    st.image('shapvalue.png')

                  #plotting graoh
def plot_shap_values_for_all(len_x, train_x, test_x, ML_model):

                random_num = random.randint(0, len_x)
                st.write(f"Displaying for row number {random_num}")
                shapValue(ML_model, train_x, test_x, 
                            tree_model=False, row_to_show=random_num)
                plt.savefig('shapvalue.png', dpi=500,
                                bbox_inches='tight')
                st.image('shapvalue.png')
                


def display_lime(train_x, train_y, test_x, feature, model):
                
                random_selector = st.button('Random_row', key="lime_001")
                if random_selector:
                    random_num = random.randint(0, len(test_x) - 1)
                    st.write(f"Displaying for row number {random_num}")
                    lime_explain(x_train=train_x.astype('float'), x_val=test_x.astype('float'),
                                        y_train = train_y.astype('float'),
                                        feat=feature, model=model, i=random_num)
                    HtmlFile = open('lime.html', 'r', encoding='utf-8')
                    source_code = HtmlFile.read()
                    components.html(source_code, height=2000)
            
def display_lime_for_all(train_X, test_X, train_y, feature, model):
            random_num = random.randint(0, len(test_X) - 1)
            st.write(f"Displaying for row number {random_num}")
            lime_explain(x_train=train_X.astype('float'), x_val=test_X.astype('float'),
                                    y_train = train_y.astype('float'),
                                    feat=feature, model=model, i=random_num)
            HtmlFile = open('lime.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=2000)

def main():
    
    #removing files already uploaded
    remove_files()

    st.set_page_config(layout='wide', page_icon='\xf0\x9f\xa7\x8a',
                   page_title='ExpainMymodel')
    
    html_txt1 = """
        ### Currently only sklearn models are compatible
    """
    html_txt2 = """<font color='blue'>Upload files to Explain</font>"""

    hide_streamlit_style = \
        '''
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
    '''
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Disable warnings

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # Select dashboard view from sidebar

    option = st.sidebar.selectbox('Select view', ('Home', 'ML Explain',
                                'Tutorial'))

    # Option to select different view of the app

    if option == 'Home':
        st.write(fixed_head, unsafe_allow_html=True)
        st.write(home_page, unsafe_allow_html=True)
        st.sidebar.markdown(overview_desc)
        demo = st.sidebar.checkbox('Video Demo')
        if demo:
            st.sidebar.video('https://res.cloudinary.com/nelsonchris/video/upload/v1631262570/explainMyModel_iyzlka.mp4', format='mp4')
        st.sidebar.markdown(about_me)
        agree = st.checkbox('Explain these methods')
        if agree:
            st.write(desc, unsafe_allow_html=True)
            

    elif option == 'ML Explain':

        st.write(fixed_head, unsafe_allow_html=True)
        st.sidebar.markdown(html_txt1, unsafe_allow_html=True)
        st.sidebar.markdown(html_txt2, unsafe_allow_html=True)
        
        demo_explain = st.sidebar.checkbox('App demo with custom dataset')

        if demo_explain:
            st.sidebar.write("Decision Tree model built to predict if a football team went home with the Man of the match award or not. The explain methods explains the contribution of each feature to the prediction")
            X_train_path = os.path.join("Classification", "train_X.csv")
            X_test_path = os.path.join("Classification", "test_X.csv")
            y_train_path = os.path.join("Classification", "y_train.csv")
            y_test_path = os.path.join("Classification", "y_test.csv")
            feat_text_path = os.path.join("Classification", "feat_text.txt")
            model_path = os.path.join("Classification", "model")


            X_train = pd.read_csv(X_train_path)
            X_test = pd.read_csv(X_test_path)
            X_len = len(X_test) -1
            y_train = pd.read_csv(y_train_path)
            y_test = pd.read_csv(y_test_path)

            with open(feat_text_path, "r") as f:
                file = f.readlines()
            contents = [feature.strip() for feature in file]

            st.write("## Permutation Importance")
            plot_perm_importance(test_x=X_test, test_y=y_test, score="accuracy",ML_model='Classification/model')

            st.write("## Partial dependency")
            read_txt_and_pdplot(feature=contents, test_X=X_test, ML_model='Classification/model')

            st.write("## SHAP")
            plot_shap_values_for_all(len_x=X_len, train_x=X_train, test_x=X_test, ML_model='Classification/model')

            st.write("## LIME")
            display_lime_for_all(train_X=X_train, train_y=y_train
                                ,test_X=X_test, feature=contents, model='Classification/model')
            
            done_explaining = st.button('Done')

            if done_explaining:
                remove_files()

            

            
        # Upload required data and model to explain

        else:
            
            train = st.sidebar.file_uploader('X_train', type=['csv', 'text'])
            if train:
                X_train = pd.read_csv(train)

            st.write("""
                
                """)

            test = st.sidebar.file_uploader('X_test', type=['csv', 'text'])
            if test is not None:
                X_test = pd.read_csv(test)
                X_len = len(X_test) - 1

            

            st.write("""
                
                """)

            train_y = st.sidebar.file_uploader('y_train', type=['csv', 'text'])
            if train_y is not None:
                y_train = pd.read_csv(train_y)

            st.write("""
                
                """)

            test_y = st.sidebar.file_uploader('y_test', type=['csv', 'text'])
            if test_y is not None:
                y_test = pd.read_csv(test_y)
            st.write("""
                
                """)

            
            model = st.sidebar.file_uploader('model')
            st.write("""
                
                """)
            
            if model is not None:
                with open('model2', 'wb') as f:
                    f.write(model.getbuffer())
            
        
            
            features = st.sidebar.file_uploader('Upload feature as txt')
            st.write("""
            
            """)
            if features:
                    stringio = StringIO(features.getvalue().decode('utf-8')) 
                    feat_col = [feature.strip() for feature in stringio.readlines()]
            
            
                

            # select if regression or classiication in order to select their evaluation metric

            

                
                


            which_ml_model = st.sidebar.selectbox('Type of ML',
                    ['Classification', 'regression'])

            # ------------------CLASSIFICATION-------------------------

            if which_ml_model == 'Classification':
                classification_score = ['accuracy', 'roc_auc', 'f1', 'precision'
                                        , 'recall']

                score = st.sidebar.selectbox('Select Classification score metric',
                                        classification_score)

                radio_option = ['None', 'Permutation Importance',
                                'Partial Density Plot','Lime', 'Shap Values', 'All']
                selected_explain = st.radio('Choose page:', radio_option)

                if selected_explain == 'Permutation Importance':

                    # compute importance and then plot perm_importance _plot

                    plot_perm_importance(test_x = X_test, test_y = y_test, score=score, ML_model='model2')
                elif selected_explain == 'Partial Density Plot':

                    # If feature.txt is uploaded, perform pdp
                    # Read feature txt file and plot pdplot.

                    read_txt_and_pdplot(feature=feat_col, test_X=X_test, ML_model='model2')
                elif selected_explain == 'Shap Values':

                # Compute and plot Shap value

                    plot_shap_values(len_x=X_len, train_x=X_train, test_x=X_test, ML_model='model2')
                
                elif selected_explain == 'Lime':

                # Compute and plot lime values

                    display_lime(train_x=X_train, test_x=X_test, train_y=y_train, feature=feat_col, model='model2')
                
                    


                elif selected_explain == 'All':

                # Display all plot
                    # compute importance and then plot perm_importance _plot

                    st.write("## Permutation Importance")

                    plot_perm_importance(test_x = X_test, test_y = y_test, score=score, ML_model='model2')

                    # Read feature txt file and plot pdplot.

                    st.write("## Partial dependency")

                    read_txt_and_pdplot(feature=feat_col, test_X=X_test, ML_model='model2')

                    # Plot shap values

                    st.write("## SHAP Value")

                    plot_shap_values_for_all(len_x=X_len, train_x=X_train, test_x=X_test, ML_model='model2')

                    # plot Lime

                    st.write("## LIME")

                    display_lime_for_all(train_X=X_train, test_X=X_test, train_y=y_train, feature=feat_col, model='model2')
                    

                else:

                    st.write('Click on any ML_explain to explain Model')
                    
                    
            # --------------RGRESSION-------------------
            else:

              #scoring the ML model

                regression_score = ['neg_mean_absolute_error',
                                    'neg_mean_squared_error', 'r2',
                                    'neg_median_absolute_error', 'max_error']
                score = st.sidebar.selectbox('Select Regression score metric',
                        regression_score)

                radio_option = ['None', 'Permutation Importance',
                                'Partial Density Plot', 'All']
                selected_explain = st.radio('Choose page:', radio_option)

                if selected_explain == 'Permutation Importance':

                    # firstly compute importance and then plot perm_importance _plot

                    plot_perm_importance(test_x = X_test, test_y = y_test, score=score, ML_model='model2')
                elif selected_explain == 'Partial Density Plot':

                    read_txt_and_pdplot(feature=feat_col, test_X=X_test, ML_model='model2')
                elif selected_explain == 'All':

                    plot_perm_importance(test_x = X_test, test_y = y_test, score=score, ML_model='model2')

                    # Read feature txt file and plot pdplot.

                    read_txt_and_pdplot(feature=feat_col, test_X=X_test, ML_model='model2')
                    
                else:

                    st.write('Click on any ML_explain to explain Model')

            done_explaining = st.button('Done')

            if done_explaining:
                remove_files()

        

        # if selected Tutorial option
    elif option == 'Tutorial':

        st.sidebar.markdown(overview_desc)
        st.write(fixed_head, unsafe_allow_html=True)
        st.write('To save column names as txt file, copy and edit This simple code'
                )
        st.code(code, language='python')

        st.write('To transform data from array to dataframe, copy and edit this simple code'
                )
        st.code(code2, language='python')

        st.write("To save model to directory.")
        st.code(code3, language='python')


if __name__ == '__main__':
    main()
