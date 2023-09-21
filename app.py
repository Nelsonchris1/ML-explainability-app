#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
# import shap


from desc import opening_explanation, fixed_head, Image, shap_image, feature_names, fixed_feature_names,\
                 feature_importance_dict, feature_importance_title #, shap_values, x_train






def read_txt_and_pdplot():
    pass


def plot_feature_importance(feature_importance_dict,n):
    top_x_features = feature_importance_dict[:n]
    top_x_features = pd.DataFrame(top_x_features).rename({0: 'Feature', 1: 'Importance'}, axis=1)



    # Create a horizontal bar chart with white bars and white axes
    fig, ax = plt.subplots(figsize=(6, 3))  # Adjust the size as needed
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    ax.barh(top_x_features['Feature'], top_x_features['Importance'], color='white')

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

    # Display the feature importances as a table
    # st.write(feature_importance_df)




# def plot_shap_values(shap_values, x_train, top_n_features):
#     shap.summary_plot(np.array(shap_values), feature_names=x_train ,plot_type='bar', plot_size=(6,3))
#     st.pyplot(plt.gcf())
#
#     print('wtf')

def plot_shap_values_for_all():
    pass
                



def main():

    st.set_page_config(layout='wide', page_icon='🤔',
                   page_title='ExpainMymodel')

    with st.sidebar:
        selected = option_menu(menu_title='Web Pro Identification',
                               options=['Main', 'Playground', 'Investigate User'],
                               icons=['house','database','file-earmark-person']
                               )

    # st.set_page_config(layout='wide', page_icon='\xf0\x9f\xa7\x8a',
    #                page_title='ExpainMymodel')



    # Select dashboard view from sidebar

    # option = st.sidebar.selectbox('Select view', ('Home', 'Playground', 'Investigate User'))
    # tab1_button = st.sidebar.button("Home", key='Home')
    # tab2_button = st.sidebar.button("Playground", key='Playground')
    # tab3_button = st.sidebar.button("Investigate User", key='Investigate User')

    # Initialize the default tab
    default_tab = 'tab1'

    # Option to select different view of the app

    if selected == 'Main':
        st.write(fixed_head, unsafe_allow_html=True)
        st.write(opening_explanation, unsafe_allow_html=True)

        st.write(feature_importance_title, unsafe_allow_html=True)

        # Create a number input for selecting the number of top features to display with custom CSS
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("<style>input[type='number'] { width: 50px !important; }</style>", unsafe_allow_html=True)


        top_n_features = st.slider('Select the number of features:', min_value=1,
                                         max_value=25, value=10, step=1)

        plot_feature_importance(feature_importance_dict, top_n_features)

        # plot_shap_values(shap_values, x_train, top_n_features)

        print('got here')



if __name__ == '__main__':
    main()
