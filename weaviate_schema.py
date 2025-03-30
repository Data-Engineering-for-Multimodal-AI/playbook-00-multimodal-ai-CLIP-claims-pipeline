import weaviate
import json

# Initialize Weaviate client with v3 syntax
client = weaviate.Client("http://localhost:8080")

# Define schema for ClaimDocument class
class_obj = {
    "class": "ClaimDocument",
    "description": "Insurance claim document with multimodal data",
    "vectorizer": "none",  # We'll provide our own vectors
    "properties": [
        {
            "name": "claim_id",
            "dataType": ["string"],
            "description": "Unique identifier for the claim"
        },
        {
            "name": "pet_name",
            "dataType": ["string"],
            "description": "Name of the pet"
        },
        {
            "name": "description",
            "dataType": ["text"],
            "description": "Claim description text"
        },
        {
            "name": "image_path",
            "dataType": ["string"],
            "description": "Path to the X-ray image file"
        },
        {
            "name": "policy_id", 
            "dataType": ["string"],
            "description": "Policy ID associated with the claim"
        },
        {
            "name": "breed",
            "dataType": ["string"],
            "description": "Pet breed"
        },
        {
            "name": "clinic",
            "dataType": ["string"],
            "description": "Veterinary clinic name"
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
            "dataType": ["string[]"],
            "description": "Historical claim data in JSON format"
        }
    ]
}

# Check if schema exists and create if not
try:
    # Check if class already exists
    schema = client.schema.get()
    existing_classes = [c["class"] for c in schema["classes"]] if "classes" in schema else []
    
    if "ClaimDocument" in existing_classes:
        print("ClaimDocument class already exists. Deleting it first.")
        client.schema.delete_class("ClaimDocument")
    
    # Create the schema
    client.schema.create_class(class_obj)
    print("Schema created successfully!")
    
except Exception as e:
    print(f"Error setting up schema: {e}")

print("Weaviate schema setup complete!") 