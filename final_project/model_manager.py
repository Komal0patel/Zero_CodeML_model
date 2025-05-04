import streamlit as st
from models.clustering.clustering import clustering_page
from models.regression.regression import regression_page
from db import log_user_action

def run_selected_model(model_name, data):
    st.sidebar.subheader("Choose Model Type")
    model_type = st.sidebar.selectbox("Model Type", ["Regression", "Classification", "Clustering"])

    if model_type == "Regression":
        regression_model = st.sidebar.selectbox("Select Regression Algorithm", [
            "Linear Regression",
            "Polynomial Regression",
            "Multiple Linear Regression",
            "Decision Tree Regression",
            "Random Forest Regression",
            "Support Vector Regression"
        ])
        # 🔍 Log before running model
        log_user_action(st.session_state.user_email, "run_model", {
            "model_type": model_type,
            "model_name": regression_model
        })
        regression_page(regression_model, data)

    elif model_type == "Clustering":
        clustering_model = st.sidebar.selectbox("Select Clustering Algorithm", [
            "K-Means",
            "DBSCAN",
            "Gaussian Mixture Model",
            "Hierarchical"
        ])
        # 🔍 Log before running model
        log_user_action(st.session_state.user_email, "run_model", {
            "model_type": model_type,
            "model_name": clustering_model
        })
        clustering_page(clustering_model, data)

    else:
        raise ValueError(f"Unsupported model: {model_name}")
