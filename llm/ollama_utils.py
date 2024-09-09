import pycurl
import json 
import os 
import requests 
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")

def accumulate_response(response):
    res = ""

    # Check if response is a string
    if isinstance(response, str):
        lines = response.split("\n")
    else:
        # Assume it's an iterable (like requests.Response)
        lines = response.iter_lines()

    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")

        if line.strip():
            try:
                chunk = json.loads(line)
                if not chunk.get("done"):
                    response_piece = chunk.get("response", "")
                    res += response_piece
                    # print(response_piece, end='', flush=True)
                else:
                    break
            except json.JSONDecodeError:
                # If it's not valid JSON, just add the line to the result
                res += line + "\n"
                print(line, end="", flush=True)

    return res


def perform_curl_request(url, data, stream=False):
    buffer = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.URL, url)
    c.setopt(c.POST, 1)
    c.setopt(c.POSTFIELDS, json.dumps(data))
    c.setopt(c.HTTPHEADER, ["Content-Type: application/json"])

    accumulated = []

    def write_callback(data):
        decoded_data = data.decode("utf-8")
        buffer.write(data)
        if stream:
            try:
                json_data = json.loads(decoded_data)
                if not json_data.get("done"):
                    response_piece = json_data.get("response", "")
                    # print(response_piece, end='', flush=True)
                    accumulated.append(response_piece)
            except json.JSONDecodeError:
                # Handle incomplete JSON data
                accumulated.append(decoded_data)

    c.setopt(c.WRITEFUNCTION, write_callback)

    try:
        c.perform()
    finally:
        c.close()

    if stream:
        return "".join(accumulated)
    else:
        return accumulate_response(buffer.getvalue().decode("utf-8"))


def perform_request(url, data, stream=False, use_pycurl=True, session=None):
    if use_pycurl:
        return perform_curl_request(url, data, stream)
    else:
        request_func = session.post if session else requests.post
        response = request_func(url, json=data, stream=stream)
        return accumulate_response(response)
    
def get_ollama_response(prompt, use_pycurl=True):
   
    data = {
                "model": "gemma2:2b",
                "prompt": prompt,
                "options": {"temperature": 0.2},
            }
    res = perform_request(
                f"{OLLAMA_URL}/api/generate",
                data,
                stream=False,
                use_pycurl=use_pycurl,
    
            )
    return res