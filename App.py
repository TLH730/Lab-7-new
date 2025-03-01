import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --------------------------
# Debug: Starting the app
# --------------------------
st.write("Starting the app...")

# --------------------------
# 1. Load and Preprocess the Data
# --------------------------
def load_data():
    file_name = 'AmesHousing.xlsx'
    st.write("Attempting to load file:", file_name)
    
    # First, try reading as an Excel workbook using openpyxl
    try:
        excel_file = pd.ExcelFile(file_name, engine="openpyxl")
        st.write("Excel file detected. Sheets found:", excel_file.sheet_names)
        if excel_file.sheet_names:
            # Read the first available worksheet
            df = pd.read_excel(file_name, sheet_name=excel_file.sheet_names[0], engine="openpyxl")
            st.write("Data loaded successfully from Excel!")
            return df
        else:
            st.write("No worksheets found in the Excel file.")
    except Exception as e:
        st.write("Error reading file as Excel:", e)
    
    # Fallback: try reading as CSV
    try:
        st.write("Attempting to read file as CSV with utf-8 encoding...")
        df = pd.read_csv(file_name)
        st.write("Data loaded successfully as CSV using utf-8!")
        return df
    except UnicodeDecodeError as ude:
        st.write("UTF-8 decoding failed, trying with latin1 encoding...")
        try:
            df = pd.read_csv(file_name, encoding="latin1")
            st.write("Data loaded successfully as CSV using latin1!")
            return df
        except Exception as e:
            st.error(f"Error reading file {file_name} as CSV with latin1 encoding: {e}")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file {file_name} as CSV: {e}")
        st.stop()

df = load_data()

st.title("Ames Housing Price Predictor")
st.write("This app predicts housing prices in Ames, Iowa based on user input features.")

# Define the required columns (adjust if your dataset structure differs)
required_columns = [
    "OverallQual",    # Overall material and finish quality
    "GrLivArea",      # Above ground living area (sq ft)
    "GarageCars",     # Number of cars that can fit in the garage
    "TotalBsmtSF",    # Total basement square feet
    "FullBath",       # Full bathrooms above grade
    "YearBuilt",      # Original construction date
    "SalePrice"       # Target variable
]

# Check if the dataset contains all required columns
if not set(required_columns).issubset(df.columns):
    st.error("The dataset is missing one or more required columns: " + ", ".join(required_columns))
    st.stop()

# Select only the required columns and drop rows with missing values
df = df[required_columns].dropna()

# --------------------------
# 2. Split the Data and Train the Regression Model
# --------------------------
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = df[features]
y = df["SalePrice"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
st.write("Model training complete.")

# --------------------------
# 3. Create the Streamlit Web App for User Input
# --------------------------
st.sidebar.header("Input House Features")

def user_input_features():
    overall_qual = st.sidebar.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=5)
    gr_liv_area = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
    garage_cars = st.sidebar.number_input("Garage Cars", min_value=0, max_value=5, value=1)
    total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_
