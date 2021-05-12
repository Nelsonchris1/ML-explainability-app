import streamlit as st
from explain import pdplot, perm_import, perm_import_plot, shapValue
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import json
from io import StringIO


html_txt = """<font color='blue'>Upload files to Explain</font>"""

st.write("Explain My")

option = st.sidebar.selectbox("Select view", ("Home", "ML Explain", "Tutorial"))
if option == "Home":
    result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://images.unsplash.com/photo-1545231027-637d2f6210f8?ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8bG9nb3xlbnwwfHwwfHw%3D&ixlib=rb-1.2.1&w=1000&q=80" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""
    prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""

    descriptive_message_temp ="""
	<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
		<h3 style="text-align:justify;color:white;padding:10px">What is Hepatitis B?</h3>
		<p style="color:white;">Hepatitis B is an infection of your liver. It’s caused by a virus. There is a vaccine that protects against it. For some people, hepatitis B is mild and lasts a short time. These “acute” cases don’t always need treatment. But it can become chronic. If that happens, it can cause scarring of the organ, liver failure, and cancer, and it even can be life-threatening.It’s spread when people come in contact with the blood, open sores, or body fluids of someone who has the hepatitis B virus.
It's serious, but if you get the disease as an adult, it shouldn’t last a long time. Your body fights it off within a few months, and you’re immune for the rest of your life. That means you can't get it again. But if you get it at birth, it’ unlikely to go away.
</p></div>
<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
<p><h3 style="color:white;" ><bold>Hepatitis B Symptoms</bold></h3>
<p style="color:white;">Short-term (acute) hepatitis B infection doesn’t always cause symptoms. For instance, it’s uncommon for children younger than 5 to have symptoms if they’re infected.
If you do have symptoms, they may include:</p>
<li style="color:white;" >Jaundice (Your skin or the whites of the eyes turn yellow, and your pee turns brown or orange.)</li>
<li style="color:white;" >Light-colored poop</li>
<li style="color:white;" >Fever</li>
<li style="color:white;" >Fatigue that persists for weeks or months</li>
<li style="color:white;" >Stomach trouble like loss of appetite, nausea, and vomiting</li>
<li style="color:white;" >Belly pain</li>
<li style="color:white;" >Joint pain</li>
<p style="color:white;">Symptoms may not show up until 1 to 6 months after you catch the virus. You might not feel anything. About a third of the people who have this disease don’t. They find out only through a blood test.
Symptoms of long-term (chronic) hepatitis B infection don’t always show up, either. If they do, they may be like those of short-term (acute) infection.
<br>
<h3><bold style="color:white;">Hepatitis B Causes and Risk Factors</bold></h3>
<p style="color:white;">It’s caused by the hepatitis B virus, and it can spread from person to person in certain ways. You can spread the hepatitis B virus even if you don’t feel sick.The most common ways to get hepatitis B include:</p>
<li style="color:white;" >Sex</li>
<li style="color:white;" >Sharing needles</li>
<li style="color:white;" >Accidental needle sticks</li>
<li style="color:white;" >Mother to child(Pregnancy)</li></p>
<p style="color:white;">The number of people who get this disease is down, the CDC says. Rates have dropped from an average of 200,000 per year in the 1980s to around 20,000 in 2016. People between the ages of 20 and 49 are most likely to get it.
About 90 of infants and 25-50 of children between the ages of 1-5 will become chronically infected. In adults, approximately 95% will recover completely and will not go on to have a chronic infection.</p>
<br/>
<br/></div>
<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
<li><a href="https://www.webmd.com/hepatitis/digestive-diseases-hepatitis-b">Learn more about Hepatitis B</a></li>    
</div>
	"""

    st.write(result_temp2, unsafe_allow_html=True)
    st.write(prescriptive_message_temp, unsafe_allow_html=True)
    st.write(descriptive_message_temp, unsafe_allow_html=True)

elif option == "ML Explain":
    st.sidebar.write(html_txt, unsafe_allow_html=True)
    train = st.sidebar.file_uploader("X_train", type=["csv", "text"])
    if train :
        X_train = pd.read_csv(train)

    st.write("""
        #
        """)

    test = st.sidebar.file_uploader("X_test", type=["csv", "text"])
    if test is not None:
        X_test = pd.read_csv(test)
        
    st.write("""
        #
        """)

    train_y = st.sidebar.file_uploader("y_train", type=["csv", "text"])
    if train_y is not None:
        y_train = pd.read_csv(train_y)
        
    st.write("""
        #
        """)

    test_y = st.sidebar.file_uploader("y_test", type=["csv", "text"])
    if test_y is not None:
        y_test = pd.read_csv(test_y)
    st.write("""
        #
        """)

    model = st.sidebar.file_uploader("model")
    st.write("""
        #
        """)

    if model is not None:
        with open(os.path.join('tempdir', 'model2'),"wb") as f: 
            f.write(model.getbuffer())

    features = st.sidebar.file_uploader("Upload feature as txt")
    st.write("""
    #
    """)
    
    which_ml_model = st.sidebar.selectbox("Type of ML", ['Classification','regression'])
    if which_ml_model == "Classification":
        classification_score = ['accuracy', 
                                'roc_auc', 'f1', 
                                'precision', 
                                'recall',
                               ]

        score = st.sidebar.selectbox("Select Classification score metric", classification_score)
    else:
        regression_score = ['neg_mean_absolute_error', 
                            'neg_mean_squared_error',  
                            'r2', 'neg_median_absolute_error', 
                            'max_error']
        score = st.sidebar.selectbox("Select Regression score metric", regression_score)

    

    # if  st.sidebar.button("Submit upload files"):
    
    radio_option = ["None","Permutation Importance","Partial Density Plot", "Shap Values", "All"]
    selected_explain = st.radio("Choose page:", radio_option)

    if selected_explain == "Permutation Importance":
         #firstly compute importance and then plot perm_importance _plot
        importances = perm_import(model='tempdir/model2', X_val=X_test, y_val=y_test, score = score, return_importances=True)
        st.dataframe(importances)
        perm_import_plot(importance=importances)
        
    elif selected_explain == "Partial Density Plot":
        if features:
            stringio = StringIO(features.getvalue().decode('utf-8'))
            feat_col = [feature.strip() for feature in stringio.readlines()]
            feat_col_name = feat_col
            feat_selected = st.selectbox("select base column name", feat_col_name)
            comapre_select = st.radio("Comapre plot", ["No", "Yes"])
            if comapre_select == "Yes":
                feat_compare_selected = st.selectbox("select coulumn to comapre", feat_col_name)
                pdplot("tempdir/model2", X_test, feat_selected)
                pdplot("tempdir/model2", X_test, feat_compare_selected, "img_pdplot2.png")
                st.image(['tempdir/img_pdplot.png', 'tempdir/img_pdplot2.png'], use_column_width=True)
            else:
                pdplot("tempdir/model2", X_test, feat_selected)
                st.image('tempdir/img_pdplot.png')

    elif selected_explain == "Shap Values":
        feat_select_shap = st.selectbox("select num of rows to predict", [0, 5, 10, 20, 30, 40, 50, 100, 200, 300])
        if feat_select_shap != 0:
            shapValue("tempdir/model2", X_train, X_test, tree_model=False, row_to_show=feat_select_shap)
            plt.savefig("tempdir/shapvalue.png",dpi=500, bbox_inches='tight')
            st.image('tempdir/shapvalue.png')

    elif selected_explain == "All":
        importances = perm_import(model='tempdir/model2', X_val=X_test, y_val=y_test, return_importances=True)
        st.dataframe(importances)
        perm_import_plot(importance=importances)

        if features:
            stringio = StringIO(features.getvalue().decode('utf-8'))
            feat_col = [feature.strip() for feature in stringio.readlines()]
            feat_col_name = feat_col
            feat_selected = st.selectbox("select base column name", feat_col_name)
            pdplot("tempdir/model2", X_test, feat_selected)
            st.image('tempdir/img_pdplot.png')

        #dispaly shap values
        shapValue("tempdir/model2", X_train, X_test, tree_model=False, row_to_show=feat_select_shap)
        plt.savefig("tempdir/shapvalue.png",dpi=500, bbox_inches='tight')
        st.image('tempdir/shapvalue.png')
    

    else:
        st.write("Click on any ML_explain to explain Model")
            