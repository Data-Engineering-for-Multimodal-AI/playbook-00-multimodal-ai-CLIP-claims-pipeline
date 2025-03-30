# Multimodal Claims Pipeline Implementation Guide - Part 2

## 3. Setting Up Weaviate Vector Database

Weaviate serves as our vector storage layer, enabling semantic search across multimodal content.

### 3.1 Start Weaviate Container

```bash
docker run -d --name weaviate -p 8080:8080 semitechnologies/weaviate:1.23.9 \
  --env QUERY_DEFAULTS_LIMIT=25 \
  --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  --env DEFAULT_VECTORIZER_MODULE=none \
  --restart unless-stopped
```

This container configuration:
- Runs Weaviate on port 8080
- Sets default query limit to 25 results
- Persists data to a Docker volume
- Disables authentication for simplicity
- Uses custom (external) vectorization

Verify Weaviate is running:

```bash
docker ps | grep weaviate
```

You should see the container status as "Up".

### 3.2 Create Weaviate Schema

Create a file `weaviate_schema.py`:

```python
import weaviate
import sys
import os

def create_schema():
    # Connect to Weaviate
    client = weaviate.Client("http://localhost:8080")
    
    # Check if Weaviate is live
    if not client.is_ready():
        print("Weaviate is not running. Please start Weaviate first.")
        sys.exit(1)
    
    # Define schema - first check if it exists
    class_obj = {
        "class": "ClaimDocument",
        "vectorizer": "none",  # We'll provide vectors from our API
        "vectorIndexType": "hnsw",
        "properties": [
            {
                "name": "claim_id",
                "dataType": ["string"],
                "description": "Unique identifier for the claim"
            },
            {
                "name": "description",
                "dataType": ["text"],
                "description": "Text description of the claim"
            },
            {
                "name": "image_path",
                "dataType": ["string"],
                "description": "Path to the associated image"
            },
            {
                "name": "pet_name", 
                "dataType": ["string"],
                "description": "Name of the pet"
            },
            {
                "name": "policy_id",
                "dataType": ["string"],
                "description": "Policy ID"
            },
            {
                "name": "breed",
                "dataType": ["string"],
                "description": "Breed of the pet"
            },
            {
                "name": "clinic",
                "dataType": ["string"],
                "description": "Clinic where treatment was provided"
            },
            {
                "name": "deductible",
                "dataType": ["number"],
                "description": "Policy deductible amount"
            },
            {
                "name": "co_pay_percent",
                "dataType": ["number"],
                "description": "Co-pay percentage"
            },
            {
                "name": "xray_covered",
                "dataType": ["boolean"],
                "description": "Whether X-rays are covered"
            },
            {
                "name": "surgery_covered",
                "dataType": ["boolean"],
                "description": "Whether surgery is covered"
            },
            {
                "name": "dental_covered",
                "dataType": ["boolean"],
                "description": "Whether dental procedures are covered"
            },
            {
                "name": "claim_history",
                "dataType": ["string"],
                "description": "JSON string of claim history events"
            }
        ]
    }
    
    # Check if class exists and delete it (for demo purposes)
    try:
        if client.schema.exists("ClaimDocument"):
            print("ClaimDocument class already exists. Deleting it first.")
            client.schema.delete_class("ClaimDocument")
        
        # Create the schema
        client.schema.create_class(class_obj)
        print("Schema created successfully!")
    except Exception as e:
        print(f"Error creating schema: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_schema()
    print("Weaviate schema setup complete!")
```

Run the schema creation script:

```bash
python weaviate_schema.py
```

Successful output should indicate "Schema created successfully!" and "Weaviate schema setup complete!".

### 3.3 Create Test Schema Data

Let's add a small utility to verify our schema:

```python
# test_weaviate.py
import weaviate

client = weaviate.Client("http://localhost:8080")
if client.is_ready():
    print("Weaviate connection successful")
    schema = client.schema.get()
    print(f"Found {len(schema['classes'])} classes")
    for cls in schema['classes']:
        print(f"Class: {cls['class']}")
        print(f"Properties: {[p['name'] for p in cls['properties']]}")
else:
    print("Failed to connect to Weaviate")
```

Run it to verify:

```bash
python test_weaviate.py
```

## 4. Building the Embedding API

Now we'll create the FastAPI service that generates vector embeddings from text and images using CLIP.

### 4.1 Create the FastAPI App

Create `app.py`:

```python
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import numpy as np

app = FastAPI(title="Claims Embedding API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int

@app.get("/")
async def root():
    return {"message": "Claims Embedding API is running"}

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(
    description: str = Form(""),
    image: Optional[UploadFile] = File(None)
):
    try:
        # Process text input
        if description:
            text_inputs = processor(
                text=[description],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        else:
            # If no description provided, use an empty string
            description = ""  # Required for when only image is provided
            text_inputs = processor(
                text=[description],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
        
        # Process image if provided
        image_inputs = None
        if image:
            image_content = await image.read()
            try:
                img = Image.open(io.BytesIO(image_content))
                image_inputs = processor(
                    images=img,
                    return_tensors="pt"
                )
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid image: {str(e)}")
        
        # Generate embeddings
        with torch.no_grad():
            if image_inputs:
                # Combined text and image embedding
                outputs = model(**text_inputs, **image_inputs)
                # Use text features as the embedding (typical approach)
                embedding = outputs.text_embeds[0].cpu().numpy().tolist()
            else:
                # Text-only embedding
                outputs = model.get_text_features(**text_inputs)
                embedding = outputs[0].cpu().numpy().tolist()
        
        # Return the embedding
        return {
            "embedding": embedding,
            "dimensions": len(embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

### 4.2 Start the Embedding API

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

The first run will download the CLIP model (~600MB). Once running, you should see:
```
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 4.3 Test the Embedding API

Create a simple test script `test_embedding_api.py`:

```python
import requests
import json

# Test text embedding
def test_text_embedding():
    url = "http://localhost:8001/embed"
    payload = {'description': 'pet with a broken leg'}
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Text embedding successful: {len(result['embedding'])} dimensions")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False

# Test image embedding
def test_image_embedding():
    url = "http://localhost:8001/embed"
    
    # You need an image file for this test
    image_path = "data/PetXrays/max_1.2.410.200028.100.3.20190927.1023440667.730.1.1.png"
    
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'description': ''}  # Empty description is required
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Image embedding successful: {len(result['embedding'])} dimensions")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    print("Testing Embedding API...")
    text_result = test_text_embedding()
    image_result = test_image_embedding()
    
    if text_result and image_result:
        print("All tests passed!")
    else:
        print("Some tests failed.")
```

Run the test:

```bash
python test_embedding_api.py
```

You should see successful generation of embeddings with 512 dimensions for both text and image inputs. 