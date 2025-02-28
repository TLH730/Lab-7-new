import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor  # Alternatively, you can use RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------
# Data Loading and Preprocessing
# ----------------------------
@st.cache  # Cache the data loading to speed up app reruns
def load_data():
    # Read the dataset from the local file in the GitHub repository
    df = pd.read_excel('AmesHousing.xlsx')  # using the provided code snippet
    # For this example, we assume 'SalePrice' is the target.
    # We select a subset of features for demonstration.
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    df = df[features + ['SalePrice']]
    # Fill missing values with the median (for numeric features)
    df.fillna(df.median(), inplace=True)
    return df

df = load_data()

# Split data into features (X) and target (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Model Training
# ----------------------------
@st.cache(allow_output_mutation=True)
def train_model():
    # Train a regression model (Linear Regression in this case)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# Optionally, you can compute and display model performance on the test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ----------------------------
# Streamlit App UI
# ----------------------------
st.title("Ames Housing Price Prediction App")
st.write("""
This app predicts the housing price based on a subset of features from the Ames Housing dataset.
""")

st.sidebar.header("Input Features")

# For each feature, we create an input widget.
def user_input_features():
    overall_qual = st.sidebar.number_input(
        "Overall Quality", 
        min_value=int(X['OverallQual'].min()), 
        max_value=int(X['OverallQual'].max()), 
        value=int(X['OverallQual'].median())
    )
    gr_liv_area = st.sidebar.number_input(
        "Above Ground Living Area (sq ft)", 
        min_value=float(X['GrLivArea'].min()), 
        max_value=float(X['GrLivArea'].max()), 
        value=float(X['GrLivArea'].median())
    )
    garage_cars = st.sidebar.number_input(
        "Garage Cars", 
        min_value=int(X['GarageCars'].min()), 
        max_value=int(X['GarageCars'].max()), 
        value=int(X['GarageCars'].median())
    )
    total_bsmt_sf = st.sidebar.number_input(
        "Total Basement Area (sq ft)", 
        min_value=float(X['TotalBsmtSF'].min()), 
        max_value=float(X['TotalBsmtSF'].max()), 
        value=float(X['TotalBsmtSF'].median())
    )
    full_bath = st.sidebar.number_input(
        "Number of Full Bathrooms", 
        min_value=int(X['FullBath'].min()), 
        max_value=int(X['FullBath'].max()), 
        value=int(X['FullBath'].median())
    )
    year_built = st.sidebar.number_input(
        "Year Built", 
        min_value=int(X['YearBuilt'].min()), 
        max_value=int(X['YearBuilt'].max()), 
        value=int(X['YearBuilt'].median())
    )

    data = {
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'GarageCars': garage_cars,
        'TotalBsmtSF': total_bsmt_sf,
        'FullBath': full_bath,
        'YearBuilt': year_built
    }
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

st.subheader("User Input Features")
st.write(input_df)

# ----------------------------
# Prediction and Output
# ----------------------------
if st.button("Predict Housing Price"):
    # Predict housing price using the trained model
    prediction = model.predict(input_df)[0]
    st.subheader("Predicted Housing Price")
    st.write(f"${prediction:,.2f}")
    st.write("Model RMSE on test set: ${:,.2f}".format(rmse))
