import streamlit as st
import os
import pandas as pd

# Custom CSS for Background and Styling
st.markdown(
    """
    <style>
        body {
            background-color: #C3A995;
            color: #593D3B;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #C3A995;
        }
        .stTextInput, .stFileUploader {
            background-color: #AB947E;
            color: #593D3B;
            border-radius: 10px;
            padding: 10px;
        }
        .stDataFrame {
            background-color: #6F5E53;
            color: white;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #8A7968;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("Upload and Preview Your Dataset")

# File Upload
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV, Excel, JSON)", 
    type=["csv", "xlsx", "json"]
)

# Create directory for uploads
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to load data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
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
    st.dataframe(data)  # Show data in a structured format
