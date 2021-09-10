#Descripion of what each explain component is 
import base64

fixed_head = """
    <h1 style="color:#d1ab69;text-align:center;letter-spacing:4px;font-family:"Times New Roman", Times, serif;" > explainMyModel </h1>
"""

Image = "ML-removebg.png"
shap_image = "SHAP.png"


home_page = f"""
    
        <div style="display:flex;justify-content:space-between;background:#01203b;padding:10px;border-radius:5px;margin:10px;">
            <div style="float:right;width:30%;background:#01203b;padding:10px;border-radius:5px;margin:10px;">
                <h3 style="color:white;letter-spacing:1px;line-height: 1.6;font-family:Arial, Helvetica, sans-serif;">
                    Designing black box machine learning algorithms are sometimes challenging and confusing to explain. 
                    But in reality, there are diffrenet ways to explain these models and also understand how each featue contributes to the accuracy of the model.
                    <br>
                    <br>
                    explainMyModel uses several model-agnostic methods to explain machine learning models by following few steps without writing code. 
                    <br><br>
                    Currently, these methods are 
                    <ol>
                        <li>Permutation Importance</li>
                        <li>Partial Dependancy plots</li>
                        <li>SHAP values</li>
                        <li>Lime</li>
                    </ol>
                </h3>
            </div>
            <div style="float:left;width:50%;background:#01203b;padding:10px;border-radius:5px;margin:10px;">
                <img style="max-height:100%;max-width:100%;" src="data:image/png;base64,{base64.b64encode(open(Image, "rb").read()).decode()}">
            </div>
        </div>

"""


descriptive_message_temp =f"""
	<div style="background-color:#044269;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h2 style="text-align:justify;color:white;padding:10px">Permutation Importance</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
		<p style="color:white;">As the name implies, Permutation feature importance describes what feature from our dataset had the most impact in building our prediction model. Permutation importance is easy to understand and also fast to process. 
        It is measured by checking how much your scoring metric reduces when each feature is missing from the dataset, This basically means that by repeatedly eliminating each feature, the model can be retrained to determine which feature is more important and which is not.</p>
        <p style="color:white;text-align:justify">But in reality retraining, the model by eliminating each feature can be computationally expensive, to avoid removing features and retraining, the values in a feature is reshuffled which implies random noise, this is how permutation feature is computed</p>
        <h3 style="text-align:justify;color:white;padding:10px">Questions it can answer</h3>
        <li style="color:white;">Which features were most important</li>
        <li style="color:white;">Which features were least important</li>
        <li style="color:white;">What is the weight of the most important feature comapared to second and third feature</li>
        <br></br>
        </div>
    <div style="background-color:#044269;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
        <h2 style="text-align:justify;color:white;padding:10px">Parital Dependancy Plot</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
        <p style="color:white;">Permutation importance indicates what feature contributed more to the model building,
         but how can we see the impact of these individual important feature. Partial dependency plot helps to visualize individual features to see their impact on the model. Partial dependency plot helps to better
        understand the relationship between predictors and the model. Knowing what feature is influencing the outcome 
        of the model is great but also  know what direction it is influencing helps for better understanding.</p>
        <h3 style="text-align:justify;color:white;padding:10px">Questions it can answer</h3>
        <li style="color:white;">What are the effect of the important feature on the predictor</li>
        <li style="color:white;">What is the target distribution as well as prediction distribution</li>
        <p></p>
        <h3 style="text-align:justify;color:white;padding:10px">NOTE!!</h3>
        <p style="color:white;">Higher value on the y axis means the corresponding value on the x axis has a greater influence on the predicting class</p>
    </div>
    <div style="background-color:#044269;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
        <h2 style="text-align:justify;color:white;padding:10px">SHAP(SHapley Additive exPlanations) values</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
        <p style="color:white;">
            Permutation importance tells us the most important and least important features, 
            Partial dependency tells us the effect of these features on the prediction class, 
            Now the SHAP value breaks downs how the model works for both individual or multiple prediction(In our case,  ROW). 
            Now we can see exactly why the outcome of a prediction is the way it is due to the help of  Shap Value.
            <br></br>
            SHAP value  explains the prediction of an instance  (row or sum of rows) by calculating the contribution of each feature from that instance to the prediction.
            Shapely values tells us how prediction is distributed to features.
            SHAP value baselines is the average of all predictions and each SHAP value i.e. their feature and corresponding value (feat_name = 2.5) is an arrow that pushes to increase or decrease its prediction. Features values causing increased predictions are in red
        <br></br>
        <img style="max-height:100%;max-width:100%;" src="data:image/png;base64,{base64.b64encode(open(shap_image, "rb").read()).decode()}">
        Here 0.7 is the output prediction while 0.4979 is the baseline value. Goal scored seems to be the biggest contribution. and Ball possession feature descreases the predicitve impact the most.
        </p>
    </div>
    <div style="background-color:#044269;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
        <h2 style="text-align:justify;color:white;padding:10px">LIME</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
        <p style="color:white;">Local Interpretable model-agnostic explanation is a machine learning explanation 
         that explains the prediction of Machine learning model locally around the prediction. 
         The technique attempts to understand the model by perturbing the input of data samples and 
         understanding how the predictions change. Lime explains the prediction of a Machine learning model 
         so that even non experts can easily understand and use them to solve business problems. 
         Lime provides a quantitative understanding between the input values and the response.  
         Lime’s output is a list of explanations reflecting the contribution of each feature to the prediction..</p>
    </div>

	"""

code = """feat_cols = train.columns
            # save columns name
with open('feat_text.txt', 'w') as f: 
        for listem in feat_cols: 
            f.write('%s backward slash N' % listem)
            #replace backward SLASH N with """

code2 = """
#Firstly save column name before transforming
feat_cols = trnain.columns

#Include transformed data and saved feature name as parameter for pandas 
train_X = pd.DataFrame(X_train, columns = feat_cols)
test_X = pd.DataFrame(X_test, columns = feat_cols)

#save both column name text and and   
with open('feat_text.txt', 'w') as f:
    f.write('%s backward slash N' % listem)
    #replaceb Backward SLASH N with
    
    # To save dataframe
train_X.to_csv("X_train.csv", index=False)
test_X.to_csv("X_test.csv", index=False)
    """

code3 = """
import pickle
# save the model to disk
filename = 'finalized_model'
pickle.dump(model, open(filename, 'wb'))
"""

overview_desc = """
    
    ------------------------
    ## Overview

    This is a web app built for easy explainability of machine learning models without writing any code in order to explain easily to non-technicals and stakeholders. 

    -------------------------
    ## Contribution

    This is an open source project and contributions will be greatly appreciated
    Github [Repo](https://github.com/Nelsonchris1/ML-explainability-app)

    1. Fork the [repo](https://github.com/Nelsonchris1/ML-explainability-app)

    2. Clone the repo

    3. Navigate to your local repository

    4. Pull latest changes from upstream to local repository

    5. create new branch

    6. Contribute

    7. Commit changes

    8. Push changes to your fork

    9. Begin pull request(PR)

    -------------------------------

     """

about_me =  """


     -------------------------------


    ## Connect with me
     [![Nelson](https://img.shields.io/badge/Author-@Nelsonchris1-gray.svg?colorA=gray&colorB=dodgergreen&logo=github)](https://www.github.com/Nelsonchris1/)

    [![Nelson3](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logoColor=white)](https://www.linkedin.com/in/nelson-ogbeide-013569171/)
    
    [![Nelson2](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)]((https://twitter.com/nelson_christof/))
    
    
    
    """