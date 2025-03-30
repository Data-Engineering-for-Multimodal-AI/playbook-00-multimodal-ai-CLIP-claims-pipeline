# Multimodal Claims Pipeline Implementation Guide - Part 3

## 5. Data Ingestion Pipeline

The next step is to build a data ingestion pipeline that extracts data from our various sources, gets embeddings from the API, and stores everything in Weaviate.

### 5.1 Create the Data Ingestion Script

Create `store_with_embedding.py`:

```python
import os
import requests
import json
import pandas as pd
import weaviate
from pathlib import Path
import sys
import time

# Configuration
EMBEDDING_API_URL = "http://localhost:8001/embed"
WEAVIATE_URL = "http://localhost:8080"
XRAY_DIR = "data/PetXrays"
CLAIM_NOTES_DIR = "data/ClaimNotes"
POLICY_CSV = "data/StructuredData/policy_details.csv"
CLAIM_HISTORY_CSV = "data/StructuredData/claim_history.csv"

def get_embedding(text=None, image_path=None):
    """Get embedding from the API for text, image or both"""
    url = EMBEDDING_API_URL
    payload = {'description': text if text else ""}
    files = {}
    
    if image_path:
        files = {'image': open(image_path, 'rb')}
    
    try:
        response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            return response.json()['embedding']
        else:
            print(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        print(f"Exception getting embedding: {e}")
        return None
    finally:
        # Close file if open
        if files and 'image' in files:
            files['image'].close()

def get_claim_from_note(filename):
    """Extract claim_id from filename and read the note content"""
    claim_id = Path(filename).stem  # Get filename without extension
    
    with open(filename, 'r') as f:
        content = f.read()
    
    return claim_id, content

def load_claim_history(claim_id):
    """Load claim history for a specific claim_id"""
    df = pd.read_csv(CLAIM_HISTORY_CSV)
    claim_history = df[df['claim_id'] == claim_id].to_dict('records')
    return json.dumps(claim_history)

def load_policy_details(policy_id):
    """Load policy details for a specific policy_id"""
    df = pd.read_csv(POLICY_CSV)
    policy = df[df['policy_id'] == policy_id]
    if not policy.empty:
        return policy.iloc[0].to_dict()
    return None

def get_policy_id_for_claim(claim_id):
    """Get policy_id associated with a claim_id"""
    df = pd.read_csv(CLAIM_HISTORY_CSV)
    claim_df = df[df['claim_id'] == claim_id]
    if not claim_df.empty:
        return claim_df.iloc[0]['policy_id']
    return None

def ingest_claim_notes():
    """Process all claim notes and ingest them into Weaviate"""
    # Connect to Weaviate
    client = weaviate.Client(WEAVIATE_URL)
    
    if not client.is_ready():
        print("Weaviate is not running. Please start Weaviate first.")
        sys.exit(1)
    
    print("Weaviate is operational")
    
    # Get all claim notes
    notes_files = [os.path.join(CLAIM_NOTES_DIR, f) for f in os.listdir(CLAIM_NOTES_DIR) 
                  if os.path.isfile(os.path.join(CLAIM_NOTES_DIR, f)) and f.endswith('.txt')]
    
    processed_count = 0
    
    for note_file in notes_files:
        # Get claim_id and content
        claim_id, content = get_claim_from_note(note_file)
        
        # Get policy_id
        policy_id = get_policy_id_for_claim(claim_id)
        if not policy_id:
            print(f"No policy found for claim {claim_id}, skipping")
            continue
        
        # Get policy details
        policy = load_policy_details(policy_id)
        if not policy:
            print(f"No policy details found for policy {policy_id}, skipping")
            continue
        
        # Get claim history
        claim_history = load_claim_history(claim_id)
        
        # Get embedding for text
        embedding = get_embedding(text=content)
        if not embedding:
            print(f"Failed to get embedding for {claim_id}, skipping")
            continue
        
        # Find matching X-ray if exists (based on pet name)
        pet_name = policy['pet_name']
        xray_files = [f for f in os.listdir(XRAY_DIR) 
                     if os.path.isfile(os.path.join(XRAY_DIR, f)) and 
                     f.lower().startswith(pet_name.lower())]
        
        image_path = None
        if xray_files:
            image_path = os.path.join(XRAY_DIR, xray_files[0])
        
        # Prepare data object
        data_object = {
            "claim_id": claim_id,
            "description": content,
            "image_path": image_path if image_path else "",
            "pet_name": pet_name,
            "policy_id": policy_id,
            "breed": policy['breed'],
            "clinic": next((ch['clinic'] for ch in json.loads(claim_history) if ch['clinic']), ""),
            "deductible": policy['deductible'],
            "co_pay_percent": policy['co_pay_percent'],
            "xray_covered": policy['xray_covered'],
            "surgery_covered": policy['surgery_covered'],
            "dental_covered": policy['dental_covered'],
            "claim_history": claim_history
        }
        
        # Store in Weaviate
        try:
            client.data_object.create(
                data_object=data_object,
                class_name="ClaimDocument",
                vector=embedding
            )
            processed_count += 1
            print(f"Processed claim {claim_id} for {pet_name}")
        except Exception as e:
            print(f"Error storing claim {claim_id} in Weaviate: {e}")
    
    print(f"Finished processing {processed_count} claims")

if __name__ == "__main__":
    ingest_claim_notes()
```

### 5.2 Run the Ingestion Script

First, make sure your API service is running:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

Then run the ingestion script:

```bash
python store_with_embedding.py
```

This will:
1. Process all claim notes in the `ClaimNotes` directory
2. Extract claim IDs and note content
3. Retrieve policy details and claim history
4. Generate embeddings for the text via the API
5. Store everything in Weaviate with the embedding

### 5.3 Verify Data Ingestion

Create a quick verification script `check_weaviate_data.py`:

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Check if Weaviate is ready
if not client.is_ready():
    print("Weaviate is not running!")
    exit(1)

# Get the class schema
schema = client.schema.get("ClaimDocument")
print(f"Schema for ClaimDocument: {schema}")

# Count the number of objects
result = client.query.aggregate("ClaimDocument").with_meta_count().do()
count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
print(f"Found {count} claims in Weaviate")

# Get a sample of the data
if count > 0:
    result = client.query.get(
        "ClaimDocument", 
        ["claim_id", "pet_name", "description"]
    ).with_limit(2).do()
    
    print("\nSample Claims:")
    for item in result["data"]["Get"]["ClaimDocument"]:
        print(f"Claim ID: {item['claim_id']}")
        print(f"Pet Name: {item['pet_name']}")
        print(f"Description: {item['description'][:100]}...")
        print()
```

Run it to verify:

```bash
python check_weaviate_data.py
```

## 6. Create a Pipeline Automation Script

Let's create a script to automate the setup and running of the pipeline:

### 6.1 Create the Run Pipeline Script

Create `run_pipeline.sh`:

```bash
#!/bin/bash

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Starting Multimodal Claims Pipeline ==="

# Check for required Python packages
echo "Checking for required Python packages..."
pip list | grep -q "fastapi" && pip list | grep -q "weaviate-client" && pip list | grep -q "streamlit"
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Missing required packages. Please run: pip install -r requirements.txt${NC}"
    exit 1
fi
echo "Dependencies look good. Continuing setup..."

# Check if Weaviate is running
echo "Checking if Weaviate is running..."
curl -s http://localhost:8080/v1/.well-known/ready > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Weaviate not running. Starting container...${NC}"
    docker run -d --name weaviate -p 8080:8080 semitechnologies/weaviate:1.23.9 \
        --env QUERY_DEFAULTS_LIMIT=25 \
        --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
        --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        --env DEFAULT_VECTORIZER_MODULE=none \
        --restart unless-stopped
    
    # Wait for Weaviate to start
    echo "Waiting for Weaviate to start..."
    for i in {1..30}; do
        curl -s http://localhost:8080/v1/.well-known/ready > /dev/null && break
        echo -n "."
        sleep 1
    done
    echo ""
    
    curl -s http://localhost:8080/v1/.well-known/ready > /dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Weaviate failed to start${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}SUCCESS: Weaviate is already running.${NC}"
fi

# Set up Weaviate schema
echo "Setting up Weaviate schema..."
python weaviate_schema.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to set up Weaviate schema${NC}"
    exit 1
fi
echo "Weaviate schema setup completed."

# Start the CLIP embedding API
echo "Starting the CLIP embedding API..."
# Check if API is already running
curl -s http://localhost:8001/ > /dev/null
if [ $? -ne 0 ]; then
    # Start the API in the background
    nohup uvicorn app:app --host 0.0.0.0 --port 8001 > embedding_api.log 2>&1 &
    API_PID=$!
    
    # Wait for API to start
    echo "Waiting for the embedding API to start..."
    for i in {1..30}; do
        curl -s http://localhost:8001/ > /dev/null && break
        echo -n "."
        sleep 1
    done
    echo ""
    
    curl -s http://localhost:8001/ > /dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Embedding API failed to start${NC}"
        cat embedding_api.log
        exit 1
    fi
fi
echo -e "${GREEN}SUCCESS: Embedding API started successfully!${NC}"

# Ingest claims data
echo "Ingesting claims data into Weaviate..."
python store_with_embedding.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to ingest claims data${NC}"
    exit 1
fi
echo -e "${GREEN}SUCCESS: Claims data ingested successfully!${NC}"

# Check data in Weaviate
echo "Checking data in Weaviate..."
python check_weaviate_data.py

# Start the Streamlit app
echo "Starting the Streamlit demo app..."
echo "Access the app at http://localhost:8501"
nohup streamlit run streamlit_app.py > streamlit.log 2>&1 &
STREAMLIT_PID=$!

# Store PIDs for cleanup
echo $API_PID > .api_pid
echo $STREAMLIT_PID > .streamlit_pid

echo -e "${GREEN}Pipeline is now running!${NC}"
echo "- Weaviate: http://localhost:8080"
echo "- Embedding API: http://localhost:8001"
echo "- Streamlit App: http://localhost:8501"
echo ""
echo "To stop the pipeline, run: ./cleanup.sh"
```

### 6.2 Create Cleanup Script

Create `cleanup.sh`:

```bash
#!/bin/bash

echo "Cleaning up..."

# Kill the API process
if [ -f .api_pid ]; then
    API_PID=$(cat .api_pid)
    kill $API_PID 2>/dev/null
    rm .api_pid
fi

# Kill the Streamlit process
if [ -f .streamlit_pid ]; then
    STREAMLIT_PID=$(cat .streamlit_pid)
    kill $STREAMLIT_PID 2>/dev/null
    rm .streamlit_pid
fi

echo "SUCCESS: Pipeline shutdown complete!"
echo "To run the app again, just run ./run_pipeline.sh"
```

Make both scripts executable:

```bash
chmod +x run_pipeline.sh cleanup.sh
```

Now you can start the entire pipeline with one command:

```bash
./run_pipeline.sh
```

And stop it with:

```bash
./cleanup.sh
``` 