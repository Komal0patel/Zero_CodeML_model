import streamlit as st
import os
import pandas as pd

# Title
st.title("Upload and Preview Your Dataset")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON)", type=["csv", "xlsx", "json"])

# Create a directory to store uploaded files
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")  # Ensure openpyxl is installed
    elif file.name.endswith(".json"):
        return pd.read_json(file)

# Display uploaded file
if uploaded_file:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved successfully: {file_path}")
    
    # Load and display data
    data = load_data(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.write(data)  # Show first few rows
