#Descripion of what each explain component is 

descriptive_message_temp ="""
	<div style="background-color:black;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h2 style="text-align:justify;color:white;padding:10px">Permutation Importance</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
		<p style="color:white;">As the name implies, Permutation feature importance describes what feature from our dataset had the most impact in building our prediction model. Permutation importance is easy to understand and also fast to process. 
        It is measured by checking how much your scoring metric reduces when each feature is missing from the dataset, This basically means that by repeatedly eliminating each feature, the model can be retrained to determine which function is more important and which is not.</p>
        <p style="color:white;text-align:justify">But in reality retraining, the model by eliminating each feature can be computationally expensive, to avoid removing features and retraining, the values in a feature is reshuffled which implies random noise, this is how permutation feature is computed</p>
        <h3 style="text-align:justify;color:white;padding:10px">Questions it can answer</h3>
        <li style="color:white;">Which features were most important</li>
        <li style="color:white;">Which features were least important</li>
        <li style="color:white;">What is the weight of the most important feature comapared to second and third feature</li>
        <br></br>
        </div>
    <div style="background-color:black;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
        <h2 style="text-align:justify;color:white;padding:10px">Parital Dependancy Plot</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
        <p style="color:white;">Permutation importance indicates what feature contributed more to the model building,
         but how can we see the impact of these individual important feature. Partial dependency plot helps to visualize individual features to see their impact on the model. Partial dependency plot helps to better
        understand the relationship between predictors and the mode. Knowing what feature is influencing the outcome 
        of the model is great but also  know what direction it is influencing helps for better understanding.</p>
        <h3 style="text-align:justify;color:white;padding:10px">Questions it can answer</h3>
        <li style="color:white;">What are the effect of the important feature on the predictor</li>
        <li style="color:white;">What is the target distribution as well as prediction distribution</li>
        <p></p>
        <h3 style="text-align:justify;color:white;padding:10px">NOTE!!</h3>
        <p style="color:white;">Higher value on the y axis means the corresponding value on the x axis has a greater influence on the predicting class</p>
    </div>
    <div style="background-color:black;overflow-x: auto; padding:10px;border-radius:10px;margin:10px;">
        <h2 style="text-align:justify;color:white;padding:10px">SHAP(SHapley Additive exPlanations) values</h2>
        <h3 style="text-align:justify;color:white;padding:10px">What it is</h3>
        <p style="color:white;">
            Permutation importance tells us the most important and least important features, 
            Partial dependency tells us the effect of these features on the prediction class, 
            Now the SHAP value breaks downs how the model works for both individual or multiple prediction(In our case,  ROW). 
            Now we can see exactly why the outcome of a prediction is the way it is due to the help of  Shap Value.
            <br></br>
            Lets use an example to explain how it works.
        We have 5 features namely Age, sex, location, salary, BuyOrNot to build a model that tells us if a client will 
        buy our goods or not. We have build our model and what to break down on why clients are likely not to buy.
        <br></br>
        Using 1 row from the dataset "[Age: 20, sex: Female, location: London, BuyOrNot: NotBuy]
        With partial dependency plot, we can see how the value age 20 impacts the prediction, 
        what if we want to see the impact of each value from all features all at once, SHAP Value gives room for this.
        </p>
    </div>

	"""

code = """feat_cols = train.columns
            # save columns name
with open('feat_text.txt', 'w') as f: 
        for listem in feat_cols: 
            f.write('%s foward slash N' % listem)
            #replace FOWARD SLASH N with /n"""

code2 = """
#Firstly save column name before transforming
feat_cols = trnain.columns

#Include transformed data and saved feature name as parameter for pandas 
train_X = pd.DataFrame(X_train, columns = feat_cols)
test_X = pd.DataFrame(X_test, columns = feat_cols)

#save both column name text and and   
with open('feat_text.txt', 'w') as f:
    f.write('%s foward slash N' % listem)
    #replace FOWARD SLASH N with /n
    
    # To save dataframe
train_X.to_csv("X_train.csv", index=False)
test_X.to_csv("X_test.csv", index=False)
    """