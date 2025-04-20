import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Yield Prediction", page_icon="🔮", layout="wide")

# Model registry
MODEL_REGISTRY = {
    "Random Forest": "models/Random Forest.pkl",
    "XGBoost": "models/XGBoost.pkl",
    "Gradient Boosting": "models/Gradient Boosting.pkl",
    "Linear Regression": "models/Linear Regression.pkl"
}

@st.cache_resource
def load_model_and_preprocessors():
    try:
        # Load model
        model_path = MODEL_REGISTRY[selected_model]
        model = joblib.load(model_path)
        
        # Load preprocessors
        scaler = joblib.load("models/scaler.pkl")
        
        # Load selected features
        with open("models/selected_features.json", "r") as f:
            selected_features = json.load(f)
        
        return model, scaler, selected_features
    except Exception as e:
        st.error(f"Error loading model and preprocessors: {str(e)}")
        return None, None, None

@st.cache_data
def load_sample_data():
    try:
        return pd.read_csv("merged_data.csv")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

st.title("🔮 Crop Yield Prediction")

# Load sample data
df = load_sample_data()
if df is None:
    st.error("Failed to load the dataset. Please check if the file exists and is accessible.")
    st.stop()

# Model selection
selected_model = st.selectbox(
    "Select Prediction Model",
    list(MODEL_REGISTRY.keys())
)

# Drop unnecessary columns as per training
cols_to_drop = ["Daily/Monthly", "Min Temperature (K)", "Max Temperature (K)", "FIPS Code",
                "reference_period_desc", "state_ansi", "county_ansi", "asd_code",
                "asd_desc", "domain_desc", "source_desc", "agg_level_desc"]

# Dictionary to store input values
input_values = {}

# Step 1: Select State
st.subheader("Step 1: Select State")
states = sorted(df["state"].unique())
selected_state = st.selectbox("State", states, key="state_selector")

# Step 2: Select County (only available after state is selected)
st.subheader("Step 2: Select County")
state_counties = sorted(df[df["state"] == selected_state]["county"].unique())
selected_county = st.selectbox("County", state_counties, key="county_selector")

# Step 3: Select Commodity (only available after county is selected)
st.subheader("Step 3: Select Commodity")
state_county_commodities = sorted(df[(df["state"] == selected_state) & (df["county"] == selected_county)]["commodity_desc"].unique())

if len(state_county_commodities) == 0:
    st.warning(f"No crop data available for {selected_county}, {selected_state}. Please select a different location.")
    selected_commodity = None
else:
    selected_commodity = st.selectbox("Commodity", state_county_commodities, key="commodity_selector")

# Step 4: Enter other features
st.subheader("Step 4: Enter Other Features")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Get min year from dataset and current year
    min_year = int(df["year"].min())
    current_year = datetime.now().year
    
    # Allow future years but not before the earliest year in the dataset
    input_values["year"] = st.number_input(
        "Year",
        min_value=min_year,
        max_value=current_year + 10,  # Allow up to 10 years in the future
        value=current_year,  # Default to current year
        help=f"Enter a year between {min_year} and {current_year + 10} for prediction"
    )
    
    input_values["Month"] = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )

with col2:
    # Get remaining numeric columns
    remaining_cols = [col for col in df.columns if col not in cols_to_drop + ["year", "Month", "state", "county", "commodity_desc", "YIELD, MEASURED IN BU / ACRE", "PRODUCTION, MEASURED IN BU"]]
    
    # Add numeric inputs
    for col in remaining_cols:
        input_values[col] = st.number_input(
            col,
            value=float(df[col].mean()),
            format="%.2f",
            help=f"Range: {df[col].min():.2f} to {df[col].max():.2f}"
        )

# Convert to categorical codes as per training
input_values["state"] = df[df["state"] == selected_state]["state"].astype("category").cat.codes.iloc[0]
input_values["county"] = df[df["county"] == selected_county]["county"].astype("category").cat.codes.iloc[0]

# Create one-hot encoded commodity features
all_commodities = sorted(df["commodity_desc"].unique())
for commodity in all_commodities:
    input_values[f"commodity_desc_{commodity}"] = 1 if commodity == selected_commodity else 0

# Submit button
submit = st.button("Predict Yield", disabled=selected_commodity is None)

if submit:
    # Load model and preprocessors
    model, scaler, selected_features = load_model_and_preprocessors()
    
    if all(x is not None for x in [model, scaler, selected_features]):
        try:
            # Create feature vector
            X = pd.DataFrame([input_values])
            
            # Get the feature names from the scaler
            scaler_feature_names = scaler.feature_names_in_
            
            # Ensure numeric features are in the same order as the scaler
            X_numeric = X[scaler_feature_names]
            
            # Scale numerical features
            X_scaled = scaler.transform(X_numeric)
            
            # Create a new DataFrame with scaled features
            X_scaled_df = pd.DataFrame(X_scaled, columns=scaler_feature_names)
            
            # Add one-hot encoded commodity features
            for commodity in all_commodities:
                X_scaled_df[f"commodity_desc_{commodity}"] = 1 if commodity == selected_commodity else 0
            
            # Select only the features used during training
            X_selected = X_scaled_df[selected_features]
            
            # Make prediction
            prediction = model.predict(X_selected)[0]
            
            # Display prediction
            st.success(f"Predicted Yield: {prediction:.2f} BU/ACRE")
            
            # Show feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(importance_df, 
                           x='Importance', 
                           y='Feature',
                           orientation='h',
                           title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Full error details:")
            st.exception(e)
    else:
        st.error("Failed to load the model or preprocessors. Please try again.") 