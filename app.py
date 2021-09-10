#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import streamlit.components.v1 as components
from explain import pdplot, perm_import, perm_import_plot, shapValue, lime_explain
from desc import descriptive_message_temp as desc
from desc import code, code2, overview_desc, home_page, fixed_head, code3, about_me
from remove import remove_files
import pandas as pd
import matplotlib.pyplot as plt
import os
import stat
import random 
import numpy as np
from io import StringIO

def main():

    remove_files()

    st.set_page_config(layout='wide', page_icon='\xf0\x9f\xa7\x8a',
                   page_title='expainMymodel')
    
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
        demo = st.sidebar.checkbox('App Demo')
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

        # Upload required data and model to explain

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

        def read_txt_and_pdplot():
                feat_selected = st.selectbox('select base column name',
                        feat_col)
                pdplot('model2', X_test, feat_selected)
                st.image('img_pdplot.png')


        def plot_perm_importance():
            importances = perm_import(model='model2',
                                    X_val=X_test, y_val=y_test,
                                    score=score, return_importances=True)
            st.dataframe(importances)
            perm_import_plot(importance=importances)


        def plot_shap_values():

            random_selector =  st.button('Random_row', key="shap_002")

            if random_selector:
                random_num = random.randint(0, X_len)
                st.write(f"Displaying for row number {random_num}")
                shapValue('model2', X_train, X_test, 
                        tree_model=False, row_to_show=random_num)
                plt.savefig('shapvalue.png', dpi=500,
                            bbox_inches='tight')
                st.image('shapvalue.png')

        def plot_shap_values_for_all():

            random_num = random.randint(0, X_len)
            st.write(f"Displaying for row number {random_num}")
            shapValue('model2', X_train, X_test, 
                        tree_model=False, row_to_show=random_num)
            plt.savefig('shapvalue.png', dpi=500,
                            bbox_inches='tight')
            st.image('shapvalue.png')
            


        def display_lime():
            
            random_selector = st.button('Random_row', key="lime_001")
            if random_selector:
                random_num = random.randint(0, X_len)
                st.write(f"Displaying for row number {random_num}")
                lime_explain(x_train=X_train.astype('float'), x_val=X_test.astype('float'),
                                    y_train = y_train.astype('float'),
                                    feat=feat_col, model='model2', i=random_num)
                HtmlFile = open('lime.html', 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=2000)
        
        def display_lime_for_all():
            random_num = random.randint(0, X_len)
            st.write(f"Displaying for row number {random_num}")
            lime_explain(x_train=X_train.astype('float'), x_val=X_test.astype('float'),
                                    y_train = y_train.astype('float'),
                                    feat=feat_col, model='model2', i=random_num)
            HtmlFile = open('lime.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            components.html(source_code, height=2000)

            
            


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

                plot_perm_importance()
            elif selected_explain == 'Partial Density Plot':

                # If feature.txt is uploaded, perform pdp
                # Read feature txt file and plot pdplot.

                read_txt_and_pdplot()
            elif selected_explain == 'Shap Values':

            # Compute and plot Shap value

                plot_shap_values()
            
            elif selected_explain == 'Lime':

            # Compute and plot lime values

               display_lime()
               
                


            elif selected_explain == 'All':

            # Display all plot
                # compute importance and then plot perm_importance _plot

                st.write("## Permutation Importance")

                plot_perm_importance()

                # Read feature txt file and plot pdplot.

                st.write("## Partial dependency")

                read_txt_and_pdplot()

                # Plot shap values

                st.write("## SHAP Value")

                plot_shap_values_for_all()

                # plot Lime

                st.write("## LIME")

                display_lime_for_all()
                

            else:

                st.write('Click on any ML_explain to explain Model')
                
                
        # --------------RGRESSION-------------------
        else:

        

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

                plot_perm_importance()
            elif selected_explain == 'Partial Density Plot':

                read_txt_and_pdplot()
            elif selected_explain == 'All':

                plot_perm_importance()

                # Read feature txt file and plot pdplot.

                read_txt_and_pdplot()
                
            else:

                st.write('Click on any ML_explain to explain Model')

        done_explaining = st.button('Done')

        if done_explaining:
            remove_files()

        

        
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