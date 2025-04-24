import streamlit as st
from PIL import Image

# Configure the main page
st.set_page_config(
    page_title="Agricultural Data Analysis",
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
    st.title("🌾 Agricultural Data Analysis and Prediction System")

st.markdown("""
Welcome to the Agricultural Data Analysis and Prediction System! This application helps you:

- 📊 Visualize agricultural data through an interactive dashboard
- 🔮 Predict crop yields using machine learning models
- 💬 Get answers to your agricultural questions through our AI assistant

Choose a page from the sidebar to get started!
""")

st.subheader("Quick Start Guide")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Dashboard
    Explore interactive visualizations of agricultural data including:
    - Crop yield trends
    - State-wise comparisons
    - Variable relationships
    """)

with col2:
    st.markdown("""
    ### 🔮 Prediction
    Predict crop yields by:
    - Selecting location
    - Choosing crop type
    - Entering environmental factors
    """)

with col3:
    st.markdown("""
    ### 💬 AI Assistant
    Get instant answers about:
    - Crop information
    - Yield predictions
    - Agricultural practices
    """) 