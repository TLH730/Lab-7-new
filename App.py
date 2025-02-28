import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------------------------
# 1. Load and Preprocess the Data
# -------------------------------------------------
@st.cache_data
def load_data():
    # Read the dataset from the local file in the GitHub repository
    df = pd.read_excel('AmesHousing.xlsx')
    return df

df = load_data()

st.title("Ames Housing Price Predictor")
st.write("This app predicts the housing price in Ames, Iowa based on user input features.")

# Define the required columns.
# (We assume these columns exist in your dataset. Adjust if necessary.)
required_columns = [
    "OverallQual",    # Overall material and finish quality
    "GrLivArea",      # Above ground living area (sq ft)
    "GarageCars",     # Number of cars that can fit in the garage
    "TotalBsmtSF",    # Total basement square feet
    "FullBath",       # Full bathrooms above grade
    "YearBuilt",      # Original construction date
    "SalePrice"       # The target variable
]

# Check if the dataset has all the required columns.
if not set(required_columns).issubset(df.columns):
    st.error(f"The dataset is missing one or more required columns: {required_columns}")
    st.stop()

# Keep only the necessary columns and drop any rows with missing values.
df = df[required_columns].dropna()

# -------------------------------------------------
# 2. Split the Data and Train the Regression Model
# -------------------------------------------------
# Define features and target
features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
X = df[features]
y = df["SalePrice"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------------------
# 3. Create the Streamlit Web App for User Input
# -------------------------------------------------
st.sidebar.header("Input House Features")

def user_input_features():
    overall_qual = st.sidebar.number_input(
        "Overall Quality (1-10)", min_value=1, max_value=10, value=5
    )
    gr_liv_area = st.sidebar.number_input(
        "Above Ground Living Area (sq ft)", min_value=300, max_value=10000, value=1500
    )
    garage_cars = st.sidebar.number_input(
        "Garage Cars", min_value=0, max_value=5, value=1
    )
    total_bsmt_sf = st.sidebar.number_input(
        "Total Basement Area (sq ft)", min_value=0, max_value=5000, value=800
    )
    full_bath = st.sidebar.number_input(
        "Full Bathrooms", min_value=0, max_value=5, value=2
    )
    year_built = st.sidebar.number_input(
        "Year Built", min_value=1872, max_value=2025, value=1970
    )

    # Create a dataframe for the input features (must match the training features)
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

# -------------------------------------------------
# 4. Make a Prediction and Display the Result
# -------------------------------------------------
prediction = model.predict(input_df)
st.subheader("Predicted Sale Price")
st.write(f"${prediction[0]:,.2f}")
