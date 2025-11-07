import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

# ---- Load model and encoders ----
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---- Streamlit UI ----
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# ---- Prepare input ----
input_data = {
    "CreditScore": [credit_score],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}

# Convert dict → DataFrame
input_data = pd.DataFrame(input_data)

# Encode gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Merge everything
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# Reorder columns to match the scaler
input_data = input_data[expected_columns]

# Scale the data
input_data_scaled = scaler.transform(input_data)

# ---- Predict ----
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# ---- Display Result ----
if prediction_proba > 0.5:
    st.success(f'The customer is likely to churn (Probability: {prediction_proba:.2f})')
else:

    st.info(f'The customer is not likely to churn (Probability: {prediction_proba:.2f})')

