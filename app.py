import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 


# Load the trained model
model = tf.keras.models.load_model('model.keras')


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
          
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)   


#customer churn prediction app

st.title('Customer Churn Prediction App')

#user input

geography = st.selectbox('Geography', ['France',' Spain', 'Germany'])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age' , 18,92)
balance = st.number_input('balance', min_value=0)
credit_score = st.slider('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])



#prepare input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    

    
})


geo_encoder = onehot_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoder, columns= onehot_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis =1)



input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]



if prediction_proba > 0.5:
    st.write('The customer is likely to churn')
    
    
else: # This was the bug, it should say "not likely to churn"
    st.write('The Customer is not likely to churn')
st.write(f'Churn Probability: {prediction_proba:.2f}')