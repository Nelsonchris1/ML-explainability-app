#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import math
from streamlit_option_menu import option_menu

from design import app_explanation, main_header, feature_importance_title, feature_playground,\
                   playground_explanation, user_investigation, investigation_explanation, feature_description_title, \
                   model_choosing_exp, features_header, features_explanation, importance_explanation, models_explanation, \
                   space_markdown, have_fun, for_more_info, features_amount_of_features, playground_amount_of_features

from model_class import models, predictions

model_name = '0 days'


def main():

    st.set_page_config(layout='wide', page_icon='🤔', page_title='ExplainMyModel')

    with st.sidebar:
        selected = option_menu(menu_title='Web Pro Identification',
                               options=['Main', 'Features', 'Playground', 'User Investigation'],
                               icons=['house', 'database', 'database', 'file-earmark-person']
                               )
        image_path = "files/ML-removebg.png"  # Change this to the actual path of your image
        st.image(image_path, caption='', width=300)


    if selected == 'Main':
        # st.write(main_header, unsafe_allow_html=True)
        #
        # st.write(models_explanation, unsafe_allow_html=True)
        #
        # st.markdown(for_more_info, unsafe_allow_html=True)
        #
        # st.write(app_explanation, unsafe_allow_html=True)
        #
        # st.write(space_markdown, unsafe_allow_html=True)
        #
        # st.write(have_fun, unsafe_allow_html=True)

        st.title("Web Pro Identification DS Model App")
        st.write("We currently have two models in production - 0 days and 7 days. "
                 "These models are meant to identify only public domain web professionals, "
                 "within 12 hours or 7 days from signup.")

        st.subheader("Model Information")
        st.markdown("For more information about how the models were built and their performance, "
                    "please [click here](#).")

        st.subheader("App Purpose")
        st.write("This app is aimed to help you understand how the models are making their predictions. Have fun!")


    if selected == 'Features':

        st.write(features_header, unsafe_allow_html=True)

        st.write(features_explanation, unsafe_allow_html=True)

        # Create a number input for selecting the number of top features to display with custom CSS
        st.markdown(
            """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        st.write(model_choosing_exp, unsafe_allow_html=True)

        colll1, colll2, colll3, colll4 = st.columns(4)

        with colll1:
            model_name = st.selectbox("", ['0 days', '7 days'], 0)
            model = models(model_name)

        st.write(features_amount_of_features, unsafe_allow_html=True)
        top_n_features = st.slider('You wil see the X most important features', min_value=1,
                                         max_value=25, value=10, step=1)


        st.write(feature_importance_title, unsafe_allow_html=True)

        st.write(importance_explanation, unsafe_allow_html=True)

        model.plot_feature_importance(top_n_features)

        st.write(feature_description_title, unsafe_allow_html=True)
        model.feature_descriptions(tab='features')

    if selected == 'Playground':

        st.write(feature_playground, unsafe_allow_html=True)

        st.write(playground_explanation, unsafe_allow_html=True)

        st.write(model_choosing_exp, unsafe_allow_html=True)
        colll1, colll2, colll3, colll4 = st.columns(4)
        with colll1:
            model_name = st.selectbox("", ['0 days', '7 days'], 0)
            model = models(model_name)

        st.write(playground_amount_of_features, unsafe_allow_html=True)
        n = st.slider('You wil see the X most important features', min_value=1,max_value=25, value=5, step=1)

        top_x_features = model.feature_importance_dict[:n]
        top_x_features = [x[0] for x in top_x_features]
        model.feature_descriptions(tab='playground')
        values = {}
        col1, col2 = st.columns(2)


        for feature in top_x_features:
            with col1:
                st.markdown("<style>input[type='number'] { width: 50px !important; }</style>", unsafe_allow_html=True)

                if type(model.unique_x_train_values[feature][0]) not in [int,float]:
                    values[feature] = st.selectbox(f"{model.feature_mapping[feature][0]}" + "-     " + f"{model.feature_mapping[feature][1]}",
                                                   sorted([x for x in model.unique_x_train_values[feature]],
                                                          reverse=False), 0)

                else:
                    min_value = min(model.unique_x_train_values[feature])
                    max_value = max(model.unique_x_train_values[feature])
                    all_values = list(range(min_value, max_value + 1))

                    values[feature] = st.selectbox(f"{model.feature_mapping[feature][0]}" + "-    " + f"{model.feature_mapping[feature][1]}",
                                                   sorted(all_values, key=lambda x: (math.isnan(x), x)),0)

        prediction = model.playground_predict(values)
        prediction = 100 * round(prediction, 3)
        predictions.append(prediction)

        with col2:
            for i in range(round(len(top_x_features)/2)):
                st.write(space_markdown, unsafe_allow_html=True)
                st.write(space_markdown, unsafe_allow_html=True)
                st.write(space_markdown, unsafe_allow_html=True)
                st.write(space_markdown, unsafe_allow_html=True)
            st.write('The predictions is the probability of the user being a web pro', unsafe_allow_html=True)


            coll1, coll2, col3 = st.columns(3)
            if len(predictions) == 1:
                coll2.metric("And the prediction is...", str(str(prediction) + '%'))
            else:
                coll2.metric("And the prediction is...", value=str(str(prediction) + '%'), delta = str(str(round(predictions[-1]-predictions[-2], 3)) + "%"))

    if selected == 'User Investigation':
        st.write(user_investigation, unsafe_allow_html=True)
        st.write(investigation_explanation, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            user_input = st.text_input("Enter uuid here:")
        if user_input:
            # Process the user input and generate a result
            uuid = f"'{user_input}'"
            model_0 = models("0 days")
            model_7 = models("7 days")

        button = st.markdown("""
                            <style>
                            div.stButton > button:first-child {
                                  background-color: rgb(108, 133, 245);
                                  height:3em; width:20%;
                                  color:white;
                            }
                            </style>""", unsafe_allow_html=True)

        go_button = st.button("Predict")
        if go_button:
            prediction_0 = model_0.investigate_predict(uuid)
            prediction_7 = model_7.investigate_predict(uuid)
            col1.metric("In the 0 days model the probability of the user being a web pro is:", str(str(100*round(prediction_0,3)) + '%'))
            col1.metric("In the 7 days model the probability of the user being a web pro is:", str(str(100 * round(prediction_7, 3)) + '%'))


if __name__ == '__main__':
    main()
