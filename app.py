import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Stroke Prediction ML App", layout="wide")

st.title("🧠 Stroke Prediction - Model Comparison App")
st.markdown("BITS M.Tech AIML - Assignment 2")

st.sidebar.header("About Project")
st.sidebar.write(
    "This application compares multiple machine learning models "
    "for predicting stroke risk based on patient health data."
)

# -------------------------
# Load Models
# -------------------------
models = {
    "Logistic Regression": joblib.load("Logistic_Regression.pkl"),
    "Decision Tree": joblib.load("Decision_Tree.pkl"),
    "KNN": joblib.load("KNN.pkl"),
    "Naive Bayes": joblib.load("Naive_Bayes.pkl"),
    "Random Forest": joblib.load("Random_Forest.pkl"),
    "XGBoost": joblib.load("XGBoost.pkl")
}

scaler = joblib.load("scaler.pkl")

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.write(data.head())
    
    # Separate features and target
    X = data.drop("stroke", axis=1)
    y = data["stroke"]
    
    # Model selection
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]
    
    # Scaling only for specific models
    if selected_model_name in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)
    
    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)
    
    st.subheader("📊 Evaluation Metrics")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"AUC: {auc:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    st.write(f"F1 Score: {f1:.3f}")
    st.write(f"MCC: {mcc:.3f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    st.pyplot(fig)
