import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import re
import joblib
from datetime import datetime
import requests
import os
from PIL import Image
import io
import base64
import chromadb
from chromadb.config import Settings
import together
import uuid

# Set page config
st.set_page_config(
    page_title="Agricultural Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize Together AI API key
together.api_key = st.secrets.get("TOGETHER_API_KEY", "")

# Initialize ChromaDB
@st.cache_resource
def init_chroma():
    try:
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chroma_db"
        ))
        return client
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return None

# Load data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("merged_data.csv")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load models
@st.cache_resource
def load_models():
    try:
        models = {}
        model_files = {
            "Random Forest": "models/Random Forest.pkl",
            "XGBoost": "models/XGBoost.pkl",
            "Gradient Boosting": "models/Gradient Boosting.pkl",
            "Linear Regression": "models/Linear Regression.pkl"
        }
        
        for name, path in model_files.items():
            try:
                models[name] = joblib.load(path)
            except:
                st.warning(f"Could not load {name} model")
        
        # Load preprocessors
        try:
            scaler = joblib.load("models/scaler.pkl")
            with open("models/selected_features.json", "r") as f:
                selected_features = json.load(f)
            return models, scaler, selected_features
        except:
            return models, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None, None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to get crop information
def get_crop_info(df, crop_name):
    crop_data = df[df['commodity_desc'].str.contains(crop_name, case=False, na=False)]
    if len(crop_data) == 0:
        return f"No information found for {crop_name}."
    
    avg_yield = crop_data['YIELD, MEASURED IN BU / ACRE'].mean()
    states = crop_data['state'].nunique()
    years = crop_data['year'].nunique()
    
    return f"Information about {crop_name}:\n- Average yield: {avg_yield:.2f} BU/ACRE\n- Grown in {states} states\n- Data available for {years} years"

# Function to get state information
def get_state_info(df, state_name):
    state_data = df[df['state'].str.contains(state_name, case=False, na=False)]
    if len(state_data) == 0:
        return f"No information found for {state_name}."
    
    crops = state_data['commodity_desc'].nunique()
    avg_yield = state_data['YIELD, MEASURED IN BU / ACRE'].mean()
    counties = state_data['county'].nunique()
    
    return f"Information about {state_name}:\n- Grows {crops} different crops\n- Average yield: {avg_yield:.2f} BU/ACRE\n- Data available for {counties} counties"

# Function to get county information
def get_county_info(df, county_name):
    county_data = df[df['county'].str.contains(county_name, case=False, na=False)]
    if len(county_data) == 0:
        return f"No information found for {county_name}."
    
    state = county_data['state'].iloc[0]
    crops = county_data['commodity_desc'].nunique()
    avg_yield = county_data['YIELD, MEASURED IN BU / ACRE'].mean()
    
    return f"Information about {county_name}, {state}:\n- Grows {crops} different crops\n- Average yield: {avg_yield:.2f} BU/ACRE"

# Function to get yield prediction
def get_yield_prediction(df, models, scaler, selected_features, crop, state, county, year, month, **kwargs):
    if not models or not scaler or not selected_features:
        return "Sorry, I couldn't load the prediction models. Please try again later."
    
    try:
        # Find the best model (Random Forest if available)
        model_name = "Random Forest" if "Random Forest" in models else list(models.keys())[0]
        model = models[model_name]
        
        # Prepare input data
        input_data = {
            "year": int(year),
            "Month": int(month),
            "state": df[df['state'] == state]['state'].astype("category").cat.codes.iloc[0],
            "county": df[df['county'] == county]['county'].astype("category").cat.codes.iloc[0]
        }
        
        # Add numeric features
        for key, value in kwargs.items():
            if key in df.columns and df[key].dtype in ['float64', 'int64']:
                input_data[key] = float(value)
        
        # Create one-hot encoded commodity features
        all_commodities = sorted(df["commodity_desc"].unique())
        for commodity in all_commodities:
            input_data[f"commodity_desc_{commodity}"] = 1 if commodity == crop else 0
        
        # Create feature vector
        X = pd.DataFrame([input_data])
        
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
            X_scaled_df[f"commodity_desc_{commodity}"] = 1 if commodity == crop else 0
        
        # Select only the features used during training
        X_selected = X_scaled_df[selected_features]
        
        # Make prediction
        prediction = model.predict(X_selected)[0]
        
        return f"Predicted yield for {crop} in {county}, {state} for {month}/{year}: {prediction:.2f} BU/ACRE"
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Function to process user query with RAG
def process_query_with_rag(query, df, models, scaler, selected_features, client):
    # First, check if it's a structured query that we can handle directly
    direct_response = process_structured_query(query, df, models, scaler, selected_features)
    if direct_response and not direct_response.startswith("I'm not sure"):
        return direct_response
    
    # If not a structured query, use RAG with Together AI
    try:
        # Get relevant context from ChromaDB
        context = get_relevant_context(query, client)
        
        # Generate response using Together AI
        response = generate_response_with_together(query, context)
        
        return response
    except Exception as e:
        st.error(f"Error in RAG processing: {str(e)}")
        return "I'm having trouble processing your query. Please try again or ask a more specific question."

# Function to process structured queries
def process_structured_query(query, df, models, scaler, selected_features):
    query = query.lower()
    
    # Check for crop information request
    crop_match = re.search(r'information about (.*?)(?:\?|$)', query)
    if crop_match:
        crop_name = crop_match.group(1).strip()
        return get_crop_info(df, crop_name)
    
    # Check for state information request
    state_match = re.search(r'information about (.*?) state(?:\?|$)', query)
    if state_match:
        state_name = state_match.group(1).strip()
        return get_state_info(df, state_name)
    
    # Check for county information request
    county_match = re.search(r'information about (.*?) county(?:\?|$)', query)
    if county_match:
        county_name = county_match.group(1).strip()
        return get_county_info(df, county_name)
    
    # Check for yield prediction request
    prediction_match = re.search(r'predict yield for (.*?) in (.*?), (.*?) for (.*?)/(.*?)(?:\?|$)', query)
    if prediction_match:
        crop = prediction_match.group(1).strip()
        county = prediction_match.group(2).strip()
        state = prediction_match.group(3).strip()
        month = prediction_match.group(4).strip()
        year = prediction_match.group(5).strip()
        
        # Get default values for other features
        default_values = {}
        numeric_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] 
                       and col not in ['YIELD, MEASURED IN BU / ACRE', 'year', 'Month', 'state', 'county']]
        
        for col in numeric_cols:
            default_values[col] = df[col].mean()
        
        return get_yield_prediction(df, models, scaler, selected_features, crop, state, county, year, month, **default_values)
    
    # Check for help request
    if 'help' in query or 'what can you do' in query:
        return """I can help you with the following:
1. Get information about a crop (e.g., "Information about corn?")
2. Get information about a state (e.g., "Information about California state?")
3. Get information about a county (e.g., "Information about Los Angeles county?")
4. Predict yield (e.g., "Predict yield for corn in Los Angeles, California for 6/2023?")
5. Show available crops, states, or counties
6. Answer general agricultural questions using AI

Just ask me a question!"""
    
    # Check for available crops, states, or counties
    if 'available crops' in query:
        crops = sorted(df['commodity_desc'].unique())
        return f"Available crops: {', '.join(crops[:10])}... (and {len(crops)-10} more)"
    
    if 'available states' in query:
        states = sorted(df['state'].unique())
        return f"Available states: {', '.join(states)}"
    
    if 'available counties' in query:
        counties = sorted(df['county'].unique())
        return f"Available counties: {', '.join(counties[:10])}... (and {len(counties)-10} more)"
    
    # Default response
    return "I'm not sure how to answer that. Try asking for information about a crop, state, or county, or ask for a yield prediction. Type 'help' for more information."

# Function to get relevant context from ChromaDB
def get_relevant_context(query, client):
    if not client:
        return "No context available."
    
    try:
        # Get the collection
        collection = client.get_or_create_collection("agricultural_knowledge")
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        # Extract and format the context
        if results and results['documents'] and results['documents'][0]:
            context = "\n\n".join(results['documents'][0])
            return context
        else:
            return "No relevant information found in the knowledge base."
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return "Error retrieving context from knowledge base."

# Function to generate response using Together AI
def generate_response_with_together(query, context):
    if not together.api_key:
        return "Together AI API key not configured. Please set up your API key."
    
    try:
        # Prepare the prompt
        prompt = f"""You are an agricultural expert assistant. Use the following context to answer the question.
If the context doesn't contain relevant information, you can provide general agricultural knowledge.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using Together AI
        response = together.Complete.create(
            prompt=prompt,
            model="togethercomputer/llama-2-70b-chat",
            max_tokens=512,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1,
        )
        
        # Extract the response text
        response_text = response['output']['choices'][0]['text']
        
        return response_text
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I'm having trouble generating a response. Please try again."

# Function to process uploaded files
def process_uploaded_file(uploaded_file, client):
    if not client:
        return "ChromaDB not initialized. Cannot process file."
    
    try:
        # Get file content
        file_content = uploaded_file.getvalue()
        
        # Process based on file type
        if uploaded_file.type == "application/pdf":
            # For PDF files, we would use a PDF parser
            # This is a placeholder - in a real implementation, you would use a PDF library
            text_content = f"Content from PDF file: {uploaded_file.name}"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # For DOCX files, we would use a DOCX parser
            # This is a placeholder - in a real implementation, you would use a DOCX library
            text_content = f"Content from DOCX file: {uploaded_file.name}"
        else:
            # For text files
            text_content = file_content.decode("utf-8")
        
        # Add to ChromaDB
        collection = client.get_or_create_collection("agricultural_knowledge")
        
        # Generate a unique ID
        doc_id = str(uuid.uuid4())
        
        # Add the document
        collection.add(
            documents=[text_content],
            ids=[doc_id],
            metadatas=[{"source": uploaded_file.name, "type": uploaded_file.type}]
        )
        
        return f"File '{uploaded_file.name}' processed and added to knowledge base."
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return f"Error processing file: {str(e)}"

# Main app
st.title("🌾 Agricultural Assistant")

# Load data and models
df = load_data()
models, scaler, selected_features = load_models()
client = init_chroma()

if df is not None:
    # Display chat interface
    st.write("Ask me anything about agricultural data, crops, or yield predictions!")
    
    # File upload for knowledge base
    with st.expander("Upload Agricultural Documents"):
        st.write("Upload PDF, DOCX, or text files to enhance the chatbot's knowledge.")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
        if uploaded_file:
            if st.button("Process File"):
                result = process_uploaded_file(uploaded_file, client)
                st.write(result)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process query and get response
        response = process_query_with_rag(prompt, df, models, scaler, selected_features, client)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
    
    # Sidebar with examples
    with st.sidebar:
        st.subheader("Example Questions")
        st.write("Try asking:")
        st.write("- Information about corn?")
        st.write("- Information about California state?")
        st.write("- Information about Los Angeles county?")
        st.write("- Predict yield for corn in Los Angeles, California for 6/2023?")
        st.write("- What crops are available?")
        st.write("- What states are available?")
        st.write("- What counties are available?")
        st.write("- How does temperature affect crop yields?")
        st.write("- What are the best practices for irrigation?")
        
        st.subheader("About")
        st.write("This chatbot uses RAG (Retrieval-Augmented Generation) with Together AI's Llama 3.2 Vision model to provide intelligent responses to agricultural questions.")
        st.write("You can upload agricultural documents to enhance the chatbot's knowledge base.")
else:
    st.error("Failed to load the dataset. Please check if the file exists and is accessible.") 