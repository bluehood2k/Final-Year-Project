import streamlit as st
from PIL import Image

# Configure the main page
st.set_page_config(
    page_title="Precision Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"  # This ensures the sidebar is visible
)

# Create two columns for layout
col1, col2 = st.columns([1, 4])

with col1:
    # Load and display the logo
    logo = Image.open("logo.png")
    st.image(logo, width=150)

with col2:
    st.title("Precision Agriculture Yield Prediction")

st.markdown("""
### Welcome to the Precision Agriculture Platform

This intelligent platform combines machine learning and agricultural expertise to help you:

- 📊 **Visualize Historical Data**: Analyze past crop performance through interactive charts
- 🔮 **Predict Crop Yields**: Use multiple ML models to forecast future yields
- 🌱 **Make Informed Decisions**: Get data-driven insights for better farming decisions

#### Available Models:
- Deep Learning (Neural Network)
- Random Forest
- XGBoost
- Gradient Boosting
- Linear Regression

Select a page from the sidebar to get started!
""") 