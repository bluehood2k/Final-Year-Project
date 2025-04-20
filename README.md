# Agricultural Data Analysis and Prediction System

A modern, professional full-stack web application that presents agricultural data insights and lets users predict crop yields using pre-trained machine learning models.

## Features

- **Interactive Dashboard**: Visualize agricultural data with interactive charts and statistics
- **Yield Prediction**: Predict crop yields based on environmental and agricultural factors
- **Agricultural Assistant**: Chatbot that answers agriculture-related queries using RAG (Retrieval-Augmented Generation)

## Project Structure

- `pages/1_📊_Dashboard.py`: Interactive dashboard with data visualizations
- `pages/2_🔮_Prediction.py`: Form for yield predictions using pre-trained models
- `pages/3_💬_Chatbot.py`: Agricultural assistant chatbot with RAG capabilities
- `models/`: Directory containing pre-trained models and preprocessors

## Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Chatbot**: Together AI (Llama 3.2), ChromaDB

## Setup and Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run Home.py`

## Usage

- **Dashboard**: Explore agricultural data through interactive visualizations
- **Prediction**: Enter feature values to predict crop yields
- **Chatbot**: Ask questions about agriculture, crops, and farming practices

## License

MIT 