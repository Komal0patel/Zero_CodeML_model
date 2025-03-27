import streamlit as st
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Title
st.title("📂 File Upload & Data Preview")

# Layout: Sidebar + Main
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "json"], help="Choose a file to upload.")

    # File Save
    UPLOAD_FOLDER = "uploaded_files"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✔ File saved successfully: {file_path}")

# Data Display Section
with col2:
    st.markdown("### Preview Uploaded Data")
    if uploaded_file:
        def load_data(file):
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                return pd.read_excel(file)
            elif file.name.endswith(".json"):
                return pd.read_json(file)
        
        data = load_data(uploaded_file)
        st.dataframe(data)
        
        # Model Selection
        st.markdown("### Select Model Type")
        model_type = st.selectbox("Choose a model:", ["Linear Regression", "Multiple Regression", "Clustering"])
        
        # Column Selection
        st.markdown("### Select Columns for Processing")
        columns = st.multiselect("Select feature columns:", data.columns)
        target_column = None
        if model_type in ["Linear Regression", "Multiple Regression"]:
            target_column = st.selectbox("Select target column:", data.columns)
        
        # Process Data
        if st.button("Run Model") and columns:
            X = data[columns]
            if model_type in ["Linear Regression", "Multiple Regression"] and target_column:
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                print(X_train)
                print(X_test)
                print(y_train)
                print(y_test)
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Multiple Regression":
                    model = LinearRegression()  # Multiple regression is handled similarly in sklearn
                
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                st.write("### Model Predictions:")
                st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": predictions}))
            
            elif model_type == "Clustering":
                k = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
                model = KMeans(n_clusters=k, random_state=42)
                data["Cluster"] = model.fit_predict(X)
                
                st.write("### Clustered Data:")
                st.dataframe(data)
