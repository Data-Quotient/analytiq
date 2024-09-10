import streamlit as st
from ollama import Client
import json

# Access configuration from Streamlit secrets
OLLAMA_URL = st.secrets["OLLAMA_URL"]
MODEL_NAME = st.secrets["MODEL_NAME"]
TEMPERATURE = float(st.secrets["TEMPERATURE"])  # Convert to float as secrets are stored as strings

def get_ollama_response(prompt):
    client = Client(host=OLLAMA_URL)
    
    # Generate a response using the Ollama client
    response = client.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature": TEMPERATURE
        }
    )
    
    # Extract the full response text
    full_response = ""
    for part in response:
        if 'response' in part:
            full_response += part['response']
    
    return full_response