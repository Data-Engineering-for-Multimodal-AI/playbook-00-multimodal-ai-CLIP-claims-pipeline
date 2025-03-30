import weaviate
import pandas as pd
import os
import requests
import json
import time
from pathlib import Path
import uuid
import sys

# Configuration
EMBEDDING_API_URL = "http://localhost:8001/embed"
WEAVIATE_URL = "http://localhost:8080"
XRAY_DIR = "data/PetXrays"
CLAIM_NOTES_DIR = "data/ClaimNotes"
POLICY_CSV = "data/StructuredData/policy_details.csv"
CLAIM_HISTORY_CSV = "data/StructuredData/claim_history.csv"

def store_claim_in_weaviate(client, claim_data, embedding, claim_id):
    """
    Store a claim document with its embedding in Weaviate
    """
    # Create data object
    try:
        # Generate a valid UUID from the claim_id string
        # Create a deterministic UUID using claim_id as the namespace
        valid_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, claim_id))
        
        data_object = {
            "claim_id": claim_id,
            "pet_name": claim_data.get("pet_name", ""),
            "description": claim_data.get("description", ""),
            "image_path": claim_data.get("image_path", ""),
            "policy_id": claim_data.get("policy_id", ""),
            "breed": claim_data.get("breed", ""),
            "clinic": claim_data.get("clinic", ""),
            "deductible": claim_data.get("deductible", 0),
            "co_pay_percent": claim_data.get("co_pay_percent", 0),
            "xray_covered": claim_data.get("xray_covered", False),
            "surgery_covered": claim_data.get("surgery_covered", False),
            "dental_covered": claim_data.get("dental_covered", False),
            "claim_history": claim_data.get("claim_history", [])
        }
        
        # Insert with vector embedding
        client.data_object.create(
            data_object=data_object,
            class_name="ClaimDocument",
            vector=embedding,
            uuid=valid_uuid  # Use valid UUID
        )
        
        print(f"Successfully stored claim {claim_id} in Weaviate")
        return True
    except Exception as e:
        print(f"Error storing claim {claim_id} in Weaviate: {e}")
        return False

def check_weaviate_running():
    """Check if Weaviate is running and ready to accept connections"""
    try:
        client = weaviate.Client("http://localhost:8080")
        schema = client.schema.get()
        print("Weaviate is operational")
        return True
    except Exception as e:
        print(f"Weaviate is not available: {e}")
        return False

def get_embedding(text=None, image_path=None):
    """Get embedding from the API for text, image or both"""
    url = EMBEDDING_API_URL
    payload = {}
    
    # Ensure description parameter is always provided (API requirement)
    if text:
        payload['description'] = text
    else:
        payload['description'] = ""  # Empty string as fallback
    
    files = {}
    
    if image_path:
        files = {'image': open(image_path, 'rb')}
    
    try:
        response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            print(f"API Response: {response.json()}")
            # Get text_embedding from the response
            return response.json()['text_embedding']
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
    # Extract claim ID from filename (format: CLM001_description.txt)
    base_name = Path(filename).stem  # Get filename without extension (CLM001_description)
    claim_id = base_name.split('_')[0]  # Extract CLM001 from the base name
    
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
    df = pd.read_csv(POLICY_CSV)
    policy = df[df['claim_id'] == claim_id]
    if not policy.empty:
        return policy.iloc[0]['policy_id']
    return None

def reset_schema():
    """Reset the Weaviate schema and create the ClaimDocument class"""
    client = weaviate.Client(WEAVIATE_URL)
    
    # Delete existing schema if it exists
    try:
        client.schema.delete_all()
        print("Existing schema deleted")
    except Exception as e:
        print(f"Error deleting schema: {e}")
    
    # Define the schema
    schema = {
        "classes": [
            {
                "class": "ClaimDocument",
                "description": "A document containing insurance claim details",
                "vectorizer": "none",  # we'll provide our own vectors
                "properties": [
                    {
                        "name": "claim_id",
                        "dataType": ["string"],
                        "description": "The unique ID of the claim"
                    },
                    {
                        "name": "claim_note",
                        "dataType": ["text"],
                        "description": "The text content of the claim note"
                    },
                    {
                        "name": "policy_id",
                        "dataType": ["string"],
                        "description": "The policy ID associated with the claim"
                    },
                    {
                        "name": "pet_name",
                        "dataType": ["string"],
                        "description": "The name of the pet"
                    },
                    {
                        "name": "breed",
                        "dataType": ["string"],
                        "description": "The breed of the pet"
                    },
                    {
                        "name": "clinic",
                        "dataType": ["string"],
                        "description": "The veterinary clinic"
                    },
                    {
                        "name": "xray_covered",
                        "dataType": ["boolean"],
                        "description": "Whether X-rays are covered by the policy"
                    },
                    {
                        "name": "surgery_covered",
                        "dataType": ["boolean"],
                        "description": "Whether surgery is covered by the policy"
                    },
                    {
                        "name": "dental_covered",
                        "dataType": ["boolean"],
                        "description": "Whether dental care is covered by the policy"
                    },
                    {
                        "name": "deductible",
                        "dataType": ["number"],
                        "description": "The deductible amount"
                    },
                    {
                        "name": "co_pay_percent",
                        "dataType": ["number"],
                        "description": "The co-pay percentage"
                    },
                    {
                        "name": "claim_history",
                        "dataType": ["text"],
                        "description": "The claim history as JSON"
                    },
                    {
                        "name": "xray_image",
                        "dataType": ["string"],
                        "description": "Path to the X-ray image file"
                    }
                ]
            }
        ]
    }
    
    # Create the schema
    try:
        client.schema.create(schema)
        print("Created schema with ClaimDocument class")
    except Exception as e:
        print(f"Error creating schema: {e}")

def ingest_claim_notes():
    """Ingest all claim notes from the data directory"""
    # Check if Weaviate is operational
    client = weaviate.Client(WEAVIATE_URL)
    if not client.is_ready():
        print("Weaviate is not operational")
        sys.exit(1)
    else:
        print("Weaviate is operational")

    # Reset schema
    reset_schema()

    # Read policy_details.csv and claim_history.csv
    policy_df = pd.read_csv("data/StructuredData/policy_details.csv")
    history_df = pd.read_csv("data/StructuredData/claim_history.csv")
    
    # Process each claim note
    claim_notes_dir = "data/ClaimNotes"
    for note_file in os.listdir(claim_notes_dir):
        if not note_file.endswith("_description.txt"):
            continue
            
        # Extract claim_id from filename
        claim_id = note_file.split("_")[0]
        
        # Read the claim note
        with open(os.path.join(claim_notes_dir, note_file), "r") as f:
            claim_note = f.read()
        
        # Get policy details for this claim
        policy_details = policy_df[policy_df["claim_id"] == claim_id]
        if policy_details.empty:
            print(f"No policy details found for {claim_id}, skipping")
            continue
            
        # Get claim history for this claim
        claim_history = history_df[history_df["claim_id"] == claim_id].to_json(orient="records")
        
        # Get pet X-ray image if available
        pet_name = policy_details.iloc[0]["pet_name"].lower()
        xray_image_path = None
        for img_file in os.listdir("data/PetXrays"):
            if pet_name in img_file.lower():
                xray_image_path = os.path.join("data/PetXrays", img_file)
                break
                
        # Get the embedding for the claim note
        embedding = get_embedding(text=claim_note)
        if embedding is None:
            print(f"Failed to get embedding for {claim_id}, skipping")
            continue
            
        # Prepare the data object
        data_object = {
            "claim_id": claim_id,
            "claim_note": claim_note,
            "policy_id": policy_details.iloc[0]["policy_id"],
            "pet_name": policy_details.iloc[0]["pet_name"],
            "breed": policy_details.iloc[0]["breed"],
            "clinic": policy_details.iloc[0]["clinic"],
            "xray_covered": bool(policy_details.iloc[0]["xray_covered"]),
            "surgery_covered": bool(policy_details.iloc[0]["surgery_covered"]),
            "dental_covered": bool(policy_details.iloc[0]["dental_covered"]),
            "deductible": float(policy_details.iloc[0]["deductible"]),
            "co_pay_percent": float(policy_details.iloc[0]["co_pay_percent"]),
            "claim_history": claim_history,
            "xray_image": xray_image_path
        }
        
        # Store in Weaviate
        try:
            client.data_object.create(
                data_object,
                "ClaimDocument",
                vector=embedding
            )
            print(f"Successfully stored {claim_id} in Weaviate")
        except Exception as e:
            print(f"Error storing {claim_id} in Weaviate: {e}")

if __name__ == "__main__":
    ingest_claim_notes() 