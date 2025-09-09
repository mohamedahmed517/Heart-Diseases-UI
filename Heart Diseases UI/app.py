import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    model = joblib.load(open("final_model.pkl", "rb"))
except:
    model = None

st.title("❤️ Heart Disease Prediction App")

st.sidebar.header("Enter Patient Data")

def user_input():
    age = st.sidebar.number_input('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting BP (trestbps)', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol (chol)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG (restecg)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate (thalach)', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.number_input('ST Depression (oldpeak)', 0.0, 6.0, 1.0, step=0.1)
    ca = st.sidebar.selectbox('No. of Major Vessels (ca)', [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
    slope_down = st.sidebar.selectbox('Slope Downsloping', [0, 1])
    slope_flat = st.sidebar.selectbox('Slope Flat', [0, 1])
    slope_up = st.sidebar.selectbox('Slope Upsloping', [0, 1])

    data = {
        'age': age, 'sex': sex, 'cp': cp,
        'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'ca': ca, 'thal': thal,
        'slope_downsloping': slope_down,
        'slope_flat': slope_flat,
        'slope_upsloping': slope_up
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

scaler = StandardScaler()

scaled_data = ['age', 'trestbps', 'chol', 'thalach']
input_df[scaled_data] = scaler.fit_transform(input_df[scaled_data])

st.subheader("User Input Data")
st.write(input_df)

if model:
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.write("1 = Heart Disease, 0 = No Heart Disease")
    st.write(f"Prediction: **{int(prediction[0])}**")

    st.subheader("Prediction Probability")
    st.write(prediction_proba)
else:
    st.warning("Model file not found. Please train and load a model.")

uploaded_file = st.file_uploader("Upload heart disease dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Heart Disease Distribution")
    fig, ax = plt.subplots()
    df['num'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("Heart Disease (num)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Age vs Cholesterol")
    fig2, ax2 = plt.subplots()
    ax2.scatter(df['age'], df['chol'], c=df['num'], cmap='viridis', alpha=0.7)
    ax2.set_xlabel("Age")
    ax2.set_ylabel("Cholesterol")

    st.pyplot(fig2)

