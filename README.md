🧠 Customer Churn Prediction using Artificial Neural Network (ANN)
📘 Overview

This project applies an Artificial Neural Network (ANN) model to predict customer churn — whether a bank customer will stay loyal or leave the bank.
By analyzing key customer details such as credit score, age, balance, and account activity, the system helps banks identify at-risk customers and take proactive steps to retain them.

🎯 Objectives

Predict if a customer is likely to churn or stay.

Build an end-to-end machine learning pipeline using TensorFlow and Scikit-learn.

Deploy an interactive Streamlit web app for real-time predictions.

🧩 Tech Stack
Category	Tools
Programming Language	Python
Deep Learning Framework	TensorFlow / Keras
Machine Learning	Scikit-learn
Data Handling	Pandas, NumPy
Web App	Streamlit
Deployment	Pyngrok (for Colab)
⚙️ Project Workflow

Data Preprocessing

Handled missing values and feature selection

Label and One-Hot Encoding for categorical features (Gender, Geography)

StandardScaler applied for feature scaling

Model Building

Built an Artificial Neural Network (ANN) using Keras Sequential API

Layers:

Input layer (based on features)

Two hidden layers with ReLU activation

Output layer with sigmoid activation

Loss function: binary_crossentropy

Optimizer: Adam

Metric: accuracy

Model Training & Evaluation

Split data into training and testing sets

Monitored model performance and prevented overfitting with EarlyStopping

Model Saving

Saved trained model as model.h5

Saved encoders and scaler as .pkl files for deployment

Streamlit App

Users can input details like credit score, age, salary, etc.

Model predicts probability of churn

Displays result as “Likely to churn” or “Not likely to churn”

Deployment on Google Colab

Streamlit app launched via pyngrok for public access

External link generated to view and test the app live

🧪 Example Prediction

Input Example:

Feature	Value
Geography	France
Gender	Male
Age	42
Balance	100000
Num of Products	2
Is Active Member	1

Output:

🟠 The customer is likely to churn (Probability: 0.76)

🗂️ Repository Structure
Customer-Churn-Prediction/
│
├── model.h5                       # Trained ANN model
├── onehot_encoder_geo.pkl         # One-hot encoder for Geography
├── label_encoder_gender.pkl       # Label encoder for Gender
├── scaler.pkl                     # StandardScaler for input normalization
├── app.py                         # Streamlit application
├── churn_dataset.csv              # Dataset used for training (if included)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies

💻 How to Run Locally

Clone this repository

git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction


Install dependencies

pip install -r requirements.txt


Run Streamlit app

streamlit run app.py


For Colab users (using ngrok)

from pyngrok import ngrok
!streamlit run app.py &
url = ngrok.connect(8501)
print(url)

📊 Results

Achieved high prediction accuracy on test data

Model successfully identifies customers likely to churn

Provides actionable insights for customer retention strategies

🙌 Acknowledgements

This project was inspired by real-world bank churn prediction problems and implemented as part of a practical exploration of ANNs in business intelligence applications.
