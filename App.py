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
    try:
        # Use ExcelFile to inspect available sheets using the openpyxl engine
        excel_file = pd.ExcelFile(file_name, engine="openpyxl")
        st.write("Available sheets in file:", excel_file.sheet_names)
        if not excel_file.sheet_names:
            st.error("No worksheets found in the Excel file. Please verify the file.")
            st.stop()
        # Read the first available worksheet
        df = pd.read_excel(file_name, sheet_name=excel_file.sheet_names[0], engine="openpyxl")
        st.write("Data loaded successfully!")
        return df
    except Exception as e:
        st.error(f"Error reading {file_name}: {e}")
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
    total_bsmt_sf = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=800)
    full_bath = st.sidebar.number_input("Full Bathrooms", min_value=0, max_value=5, value=2)
    year_built = st.sidebar.number_input("Year Built", min_value=1872, max_value=2025, value=1970)
    
    # Build a DataFrame from the user inputs matching the training features
    data = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "YearBuilt": year_built,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("Input Features")
st.write(input_df)

# --------------------------
# 4. Make a Prediction and Display the Result
# --------------------------
prediction = model.predict(input_df)
st.subheader("Predicted Sale Price")
st.write(f"${prediction[0]:,.2f}")
