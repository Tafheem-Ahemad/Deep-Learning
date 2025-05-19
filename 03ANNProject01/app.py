import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
# import list

# import all models
with open('one_hot_encoder_Gender.pkl','rb') as file:
	one_hot_encoder_Gender=pickle.load(file)

with open('one_hot_encoder_Geography.pkl','rb') as file:
	one_hot_encoder_Geography=pickle.load(file)

with open('scalar.pkl','rb') as file:
	scalar=pickle.load(file)

model=load_model("model.h5")


## streamlit app
st.title('Customer Churn Prediction')

# Select the values
Country=st.selectbox("Select your Country",list(one_hot_encoder_Geography.categories_[0]))
Gender=st.selectbox("Select your gender",list(one_hot_encoder_Gender.categories_[0]))
Age=st.slider("Select your age",min_value=0,max_value=100)
HasCrCard=st.selectbox("Have you a credit card",["YES","NO"])
IsActiveMember=st.selectbox("Is your credit card is Active",["YES","NO"])
CreditScore=st.number_input("Enter your Credit score , if you have a Credit card and It active active")
Salary=st.number_input("Enter your salary")
Balance=st.number_input("Enter your balance")
Tenure=st.slider("Enter your Tenure",min_value=1,max_value=4)
NumOfProducts=st.slider("Enter number of products",min_value=1,max_value=3)

# Create the Input dataframe
input_data = {
    'CreditScore': [CreditScore],
    'Geography': [Country],
    'Gender': [Gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': Balance,
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [1 if HasCrCard == "YES" else 0],
    'IsActiveMember': [1 if IsActiveMember == "YES" else 0],
    'EstimatedSalary': [Salary]
}

input_df=pd.DataFrame(input_data)

# Encode the classification data
gender_encoder=one_hot_encoder_Gender.transform(input_df[['Gender']]).toarray()
geography_encoder=one_hot_encoder_Geography.transform(input_df[['Geography']]).toarray()

input_df_encoded=pd.concat([
	input_df.drop(columns=['Gender','Geography'],axis=1),
	pd.DataFrame(gender_encoder,columns=one_hot_encoder_Gender.get_feature_names_out()),
	pd.DataFrame(geography_encoder,columns=one_hot_encoder_Geography.get_feature_names_out())

],axis=1)

# Scandardscale the data
input_df_scaled=scalar.transform(input_df_encoded)

prediction=model.predict(input_df_scaled)
probaliblity_srcore=prediction[0][0]

# print the output
st.write(f'Churn Probability: {probaliblity_srcore}')

if probaliblity_srcore > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
	


