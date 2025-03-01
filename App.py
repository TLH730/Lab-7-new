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

st.write("### Dataset Preview")
st.write(df.head())

# Display all column names for debugging purposes
st.write("### Column Names in the Dataset")
st.write(df.columns.tolist())

# For simplicity, drop rows with missing values.
df = df.dropna()

# --------------------------
# Feature Selection and Splitting
# --------------------------
# Ensure that the target variable exists.
if 'SalePrice' not in df.columns:
    st.error("Error: The dataset does not contain a 'SalePrice' column.")
    st.stop()

# Define a subset of features to use in the model.
# Adjust these based on the actual column names you see in the output above.
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

# Filter the selected features to only include those that exist in the dataset.
features = [feat for feat in selected_features if feat in df.columns]

if not features:
    st.error("Error: None of the selected features exist in the dataset. Please update the feature list based on the available columns.")
    st.stop()

X = df[features]
y = df['SalePrice']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Model Training
# --------------------------
# Create a pipeline that scales the features then applies Linear Regression.
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
    for feature in features:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())
        # Create a slider for each feature.
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
