"""
Simple test app to verify the pipeline components are working correctly.
This script checks:
1. Streamlit app is running
2. Embedding API is running
3. Can successfully get embeddings from text/images
"""

import os
import requests
import json
from pathlib import Path

# Configuration
EMBEDDING_API_URL = "http://localhost:8001/embed"
STREAMLIT_URL = "http://localhost:8501"
SAMPLE_XRAY = "data/PetXrays/emma_1.2.410.200028.100.3.20190924.1430080534.21770.1.1.png"
GREEN = '\033[0;32m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color


def test_streamlit_app():
    """Test if Streamlit app is running"""
    try:
        response = requests.get(STREAMLIT_URL, timeout=5)
        if response.status_code == 200:
            print(f"{GREEN}✅ Streamlit app is running{NC}")
            return True
        else:
            print(f"{RED}❌ Streamlit app returned status code {response.status_code}{NC}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{RED}❌ Streamlit app is not running: {e}{NC}")
        print(f"   Make sure to start it with: streamlit run streamlit_app.py")
        return False


def test_embedding_api():
    """Test if embedding API is running"""
    try:
        response = requests.get("http://localhost:8001/", timeout=5)
        if response.status_code == 200:
            print(f"{GREEN}✅ Embedding API is running{NC}")
            return True
        else:
            print(f"{RED}❌ Embedding API returned status code {response.status_code}{NC}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{RED}❌ Embedding API is not running: {e}{NC}")
        print(f"   Make sure to start it with: uvicorn app:app --host 0.0.0.0 --port 8001")
        return False


def test_text_embedding():
    """Test text embedding"""
    test_text = "cough"
    files = {}
    data = {"description": test_text}
    
    try:
        response = requests.post(EMBEDDING_API_URL, files=files, data=data)
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            if embedding and len(embedding) > 0:
                print(f"{GREEN}✅ Successfully got embedding for '{test_text}'{NC}")
                print(f"{GREEN}✅ Embedding received with length: {len(embedding)}{NC}")
                return True
            else:
                print(f"{RED}❌ Embedding API returned empty embedding{NC}")
                return False
        else:
            print(f"{RED}❌ Failed to get embedding for text: {response.status_code} - {response.text}{NC}")
            return False
    except Exception as e:
        print(f"{RED}❌ Error while getting text embedding: {e}{NC}")
        return False


def test_image_embedding():
    """Test image embedding"""
    if not os.path.exists(SAMPLE_XRAY):
        print(f"{RED}❌ Sample X-ray image not found at {SAMPLE_XRAY}{NC}")
        return False
    
    files = {"image": open(SAMPLE_XRAY, "rb")}
    data = {}
    
    try:
        response = requests.post(EMBEDDING_API_URL, files=files, data=data)
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            if embedding and len(embedding) > 0:
                print(f"{GREEN}✅ Successfully got embedding for X-ray image{NC}")
                print(f"{GREEN}✅ Image embedding received with length: {len(embedding)}{NC}")
                return True
            else:
                print(f"{RED}❌ Embedding API returned empty embedding for image{NC}")
                return False
        else:
            print(f"{RED}❌ Failed to get embedding for image: {response.status_code} - {response.text}{NC}")
            return False
    except Exception as e:
        print(f"{RED}❌ Error while getting image embedding: {e}{NC}")
        return False
    finally:
        # Close the file
        files["image"].close()


def test_combined_embedding():
    """Test combined text + image embedding"""
    if not os.path.exists(SAMPLE_XRAY):
        print(f"{RED}❌ Sample X-ray image not found at {SAMPLE_XRAY}{NC}")
        return False
    
    test_text = "cough symptoms"
    files = {"image": open(SAMPLE_XRAY, "rb")}
    data = {"description": test_text}
    
    try:
        response = requests.post(EMBEDDING_API_URL, files=files, data=data)
        if response.status_code == 200:
            embedding = response.json().get("embedding")
            if embedding and len(embedding) > 0:
                print(f"{GREEN}✅ Successfully got combined embedding{NC}")
                print(f"{GREEN}✅ Combined embedding received with length: {len(embedding)}{NC}")
                return True
            else:
                print(f"{RED}❌ Embedding API returned empty combined embedding{NC}")
                return False
        else:
            print(f"{RED}❌ Failed to get combined embedding: {response.status_code} - {response.text}{NC}")
            return False
    except Exception as e:
        print(f"{RED}❌ Error while getting combined embedding: {e}{NC}")
        return False
    finally:
        # Close the file
        files["image"].close()


def main():
    streamlit_running = test_streamlit_app()
    embedding_api_running = test_embedding_api()
    
    if embedding_api_running:
        text_embedding_ok = test_text_embedding()
        image_embedding_ok = test_image_embedding()
        combined_embedding_ok = test_combined_embedding()
    
    print("\n=== Streamlit App Test Summary ===")
    if streamlit_running:
        print(f"The Streamlit app is running at {STREAMLIT_URL}")
    else:
        print(f"The Streamlit app is NOT running at {STREAMLIT_URL}")
    
    if embedding_api_running:
        print(f"The embedding API is running at http://localhost:8001")
    else:
        print(f"The embedding API is NOT running")
    
    print("Manual tests of the web interface recommended for interactive elements")


if __name__ == "__main__":
    main() 