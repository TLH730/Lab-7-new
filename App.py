import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# App title
st.title("Ames Housing Price Predictor")

# --------------------------
# Data Loading and Preprocessing
# --------------------------
@st.cache_data
def load_data():
    # Read the dataset from the local file in the GitHub repository
    df = pd.read_excel('AmesHousing (1).xlsx')
    # Remove extra spaces from column names if any
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Display initial dataset information for debugging
st.write("### Dataset Preview")
st.write(df.head())
st.write("### Dataset Shape Before Cleaning:", df.shape)

# Define a subset of features with exact column names.
selected_features = [
    'Overall Qual',   # Correct column name
    'Gr Liv Area',    # Correct column name
    'Garage Cars',    # Correct column name
    'Total Bsmt SF',  # Correct column name
    'Full Bath',      # Correct column name
    'Year Built'      # Correct column name
]

# Drop missing values only for the selected features and target
df = df.dropna(subset=selected_features + ['SalePrice'])
st.write("### Dataset Shape After Cleaning:", df.shape)

# Check if the dataset is empty after cleaning
if df.empty:
    st.error("Dataset is empty after cleaning. Please check your data or consider imputing missing values instead of dropping them.")
    st.stop()

# --------------------------
# Feature Selection and Splitting
# --------------------------
if 'SalePrice' not in df.columns:
    st.error("Error: The dataset does not contain a 'SalePrice' column.")
    st.stop()

X = df[selected_features]
y = df['SalePrice']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Model Training
# --------------------------
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

st.write("### Model Training Completed")
st.write("The regression model has been trained on the Ames Housing dataset.")

# --------------------------
# Sidebar: User Input for Prediction
# --------------------------
st.sidebar.header("Input Features for Prediction")

def user_input_features():
    input_data = {}
    for feature in selected_features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        input_data[feature] = st.sidebar.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    return pd.DataFrame([input_data])

input_df = user_input_features()

st.write("### User Input Features")
st.write(input_df)

# --------------------------
# Prediction
# --------------------------
prediction = model.predict(input_df)
st.write("### Predicted Sale Price")
st.write(f"${prediction[0]:,.2f}")
