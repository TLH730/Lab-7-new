import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------
# Data Loading and Preprocessing
# ----------------------------
@st.cache
def load_data():
    try:
        # Specify the sheet name and engine; adjust 'Sheet1' as needed
        df = pd.read_excel('AmesHousing.xlsx', sheet_name='Sheet1', engine='openpyxl')
        # Ensure the necessary columns exist
        required_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in the Excel file.")
                return None
        # Select a subset of features for demonstration
        features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        df = df[features + ['SalePrice']]
        # Fill missing values with the median value of each column
        df.fillna(df.median(), inplace=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    return df

df = load_data()

if df is not None:
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
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    model = train_model()

    # Evaluate model performance
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # ----------------------------
    # Streamlit App UI
    # ----------------------------
    st.title("Ames Housing Price Prediction App")
    st.write("This app predicts the housing price based on selected features from the Ames Housing dataset.")

    st.sidebar.header("Input Features")
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
        prediction = model.predict(input_df)[0]
        st.subheader("Predicted Housing Price")
        st.write(f"${prediction:,.2f}")
        st.write("Model RMSE on test set: ${:,.2f}".format(rmse))
else:
    st.error("Data could not be loaded. Please check the Excel file and its contents.")
