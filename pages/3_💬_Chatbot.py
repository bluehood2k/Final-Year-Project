__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import re
import os
from PIL import Image
import io
import base64
import chromadb
from chromadb.config import Settings
from together import Together
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Together as LangchainTogether
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Agricultural Assistant",
    page_icon="💬",
    layout="wide"
)

# Initialize Together AI client
together_client = Together(api_key=os.getenv("TOGETHER_API_KEY", ""))

# Initialize LangChain memory and conversation chain
@st.cache_resource
def init_conversation():
    try:
        # Initialize the LLM
        llm = LangchainTogether(
            model="meta-llama/Llama-Vision-Free",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1,
        )

        # Create a custom prompt template
        template = """You are an agricultural assistant that helps users understand crop yields, make predictions, and analyze agricultural data. 
You should ONLY answer questions related to agriculture, farming, crops, livestock, and food production.
If a question is not related to agriculture or farming, politely inform the user that you can only answer questions about agriculture and farming.

Current conversation:
{history}

Human: {input}
Assistant: Let me help you with that."""

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )

        # Initialize memory
        memory = ConversationBufferMemory(return_messages=True)

        # Create conversation chain
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )

        return conversation
    except Exception as e:
        st.error(f"Error initializing conversation: {str(e)}")
        return None

# Initialize ChromaDB with the new configuration
@st.cache_resource
def init_chroma():
    try:
        # Create the directory if it doesn't exist
        os.makedirs("chroma_db", exist_ok=True)
        
        # Using the new client configuration
        client = chromadb.PersistentClient(path="chroma_db")
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

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

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

# Function to process user query
def process_query(query, df):
    # First, check if it's a structured query that we can handle directly
    direct_response = process_structured_query(query, df)
    if direct_response and not direct_response.startswith("I'm not sure"):
        return direct_response
    
    # If not a structured query, use the Together API directly with conversation history
    try:
        # Prepare messages with conversation history
        messages = []
        
        # Add system message to instruct the model
        messages.append({
            "role": "system", 
            "content": "You are an agricultural assistant that helps users understand crop yields, make predictions, and analyze agricultural data. You should ONLY answer questions related to agriculture, farming, crops, livestock, and food production. If a question is not related to agriculture or farming, DO NOT answer the question and politely inform the user that you can only answer questions about agriculture and farming."
        })
        
        # Add conversation history (up to the last 5 exchanges to keep context manageable)
        for i in range(max(0, len(st.session_state.conversation_history) - 10), len(st.session_state.conversation_history), 2):
            if i + 1 < len(st.session_state.conversation_history):
                messages.append({"role": "user", "content": st.session_state.conversation_history[i]})
                messages.append({"role": "assistant", "content": st.session_state.conversation_history[i+1]})
        
        # Add the current query
        messages.append({"role": "user", "content": query})
        
        # Use the Together API with conversation history
        response = together_client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=messages
        )
        
        # Get the response
        response_text = response.choices[0].message.content
        
        # Update conversation history
        st.session_state.conversation_history.append(query)
        st.session_state.conversation_history.append(response_text)
        
        return response_text
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I'm having trouble processing your query. Please try again or ask a more specific question."

# Function to process structured queries
def process_structured_query(query, df):
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
        
        return f"I can help you predict the yield for {crop} in {county}, {state} for {month}/{year}. Please use the Prediction page for detailed yield predictions with all available features."
    
    return None

# Main app
def main():
    st.title("Agricultural Assistant")
    
    # Load data
    df = load_data()
    client = init_chroma()
    
    if df is None:
        st.error("Error loading data. Please check the console for details.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This agricultural assistant can help you with:
        - Getting information about crops, states, and counties
        - Answering general agricultural questions
        - Providing guidance on using the prediction page
        """)
        
        st.header("Example Questions")
        st.write("""
        - "What's the average yield for corn in Iowa?"
        - "Tell me about California's agricultural production"
        - "What crops are grown in Texas?"
        - "How can I predict crop yields?"
        - "What are some easy crops to grow as a beginner?"
        """)
        
        # Add a button to clear conversation history
        if st.button("Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.messages = []
            st.success("Conversation history cleared!")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know about agriculture?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            response = process_query(prompt, df)
            st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 