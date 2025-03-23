import streamlit as st
import os
import pandas as pd



# Title
st.title("📂 File Upload & Data Preview")
# Layout: Sidebar + Main
col1, col2 = st.columns([1,2])

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
