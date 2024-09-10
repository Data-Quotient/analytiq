import streamlit as st
from ollama import Client

# Access configuration from Streamlit secrets
OLLAMA_URL = st.secrets["OLLAMA_URL"]
MODEL_NAME = st.secrets["MODEL_NAME"]
TEMPERATURE = float(st.secrets["TEMPERATURE"])  # Convert to float as secrets are stored as strings

def get_ollama_response(prompt):
    client = Client(host=OLLAMA_URL)
    
    # Generate a response using the Ollama client
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": TEMPERATURE
        }
    )
    
    # The response is now a dictionary, we can directly access the content
    return response['message']['content']