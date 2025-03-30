import requests
import os
import sys
import json
import weaviate
from PIL import Image
import io
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
XRAYS_DIR = DATA_DIR / "PetXrays"
PET_PHOTOS_DIR = DATA_DIR / "PetPhotos"
EMBEDDING_API_URL = "http://localhost:8001/embed"
WEAVIATE_URL = "http://localhost:8080"

def print_header(text):
    """Print a formatted header for test sections"""
    print("\n" + "="*80)
    print(f"TEST: {text}")
    print("="*80)

def print_result(pet_name, found, source):
    """Print test result with consistent formatting"""
    result = "✅ PASS" if found else "❌ FAIL"
    print(f"{result}: Found {pet_name} using {source}")

def connect_to_weaviate():
    """Connect to Weaviate database"""
    try:
        client = weaviate.Client(WEAVIATE_URL)
        if client.is_ready():
            print("✅ Connected to Weaviate")
            return client
        else:
            print("❌ Weaviate is not ready")
            return None
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return None

def check_api_status():
    """Check if the embedding API is running"""
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            print("✅ Embedding API is running")
            return True
        else:
            print(f"❌ Embedding API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Embedding API: {e}")
        return False

def get_claim_count(client):
    """Get the number of claims in the database"""
    try:
        result = client.query.aggregate("ClaimDocument").with_meta_count().do()
        count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
        print(f"📊 Found {count} claims in the database")
        return count
    except Exception as e:
        print(f"❌ Failed to get claim count: {e}")
        return 0

def get_embedding(text=None, image_path=None):
    """Get embedding from the API for text and/or image"""
    url = EMBEDDING_API_URL
    
    if not text and not image_path:
        print("❌ Error: Please provide either text or an image path")
        return None
        
    files = {}
    data = {"description": text if text else ""}
    
    if image_path:
        try:
            files = {"image": open(image_path, "rb")}
        except Exception as e:
            print(f"❌ Error opening image file {image_path}: {e}")
            return None
    
    try:
        print(f"📤 Sending request to embedding API with text='{text}', image={'Yes' if image_path else 'No'}")
        response = requests.post(url, files=files, data=data)
        
        if files and "image" in files:
            files["image"].close()
            
        if response.status_code == 200:
            result = response.json()
            embedding = result.get("embedding")
            if embedding:
                print(f"✅ Successfully received embedding with {len(embedding)} dimensions")
                return embedding
            else:
                print(f"❌ No embedding found in API response: {result}")
                return None
        else:
            print(f"❌ Error from embedding API: {response.status_code}")
            print(f"Response content: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception when getting embedding: {e}")
        return None

def search_by_vector(client, embedding, pet_name_to_find=None, limit=5):
    """Search for claims using vector similarity"""
    try:
        vector_query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_near_vector({
            "vector": embedding,
            "certainty": 0.5  # Use a lower threshold for testing
        }).with_limit(limit).do()
        
        if "data" in vector_query and "Get" in vector_query["data"] and "ClaimDocument" in vector_query["data"]["Get"]:
            results = vector_query["data"]["Get"]["ClaimDocument"]
            
            # Print the results
            print(f"📋 Found {len(results)} results with vector search")
            for i, result in enumerate(results):
                print(f"  Result #{i+1}: {result['pet_name']} - {result['claim_id']}")
                
            # Check if the pet we're looking for is in the results
            if pet_name_to_find:
                for result in results:
                    if result["pet_name"].lower() == pet_name_to_find.lower():
                        return True, results
                return False, results
            return True, results
        else:
            print("❌ No results found with vector search")
            return False, []
    except Exception as e:
        print(f"❌ Error in vector search: {e}")
        return False, []

def search_by_text(client, query_text, pet_name_to_find=None, limit=5):
    """Search for claims using text matching"""
    try:
        # Create a general search query across multiple fields
        text_query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_where({
            "operator": "Or",
            "operands": [
                {
                    "path": ["claim_note"],
                    "operator": "Like",
                    "valueText": f"*{query_text}*"
                },
                {
                    "path": ["pet_name"],
                    "operator": "Like",
                    "valueText": f"*{query_text}*"
                }
            ]
        }).with_limit(limit).do()
        
        if "data" in text_query and "Get" in text_query["data"] and "ClaimDocument" in text_query["data"]["Get"]:
            results = text_query["data"]["Get"]["ClaimDocument"]
            
            # Print the results
            print(f"📋 Found {len(results)} results with text search")
            for i, result in enumerate(results):
                print(f"  Result #{i+1}: {result['pet_name']} - {result['claim_id']}")
                
            # Check if the pet we're looking for is in the results
            if pet_name_to_find:
                for result in results:
                    if result["pet_name"].lower() == pet_name_to_find.lower():
                        return True, results
                return False, results
            return True, results
        else:
            print("❌ No results found with text search")
            return False, []
    except Exception as e:
        print(f"❌ Error in text search: {e}")
        return False, []

def test_text_search(client):
    """Test searching by text keywords"""
    print_header("TEXT SEARCH")
    
    # Test searching for each pet by name
    test_pets = [
        ("Emma", "emma"),
        ("Jack", "jack"),
        ("Max", "max"),
        ("Otis", "otis"),
        ("Rocky", "rocky")
    ]
    
    all_passed = True
    
    for display_name, search_name in test_pets:
        print(f"\n🔍 Testing text search for '{display_name}'")
        success, _ = search_by_text(client, search_name, display_name)
        print_result(display_name, success, "text search")
        all_passed = all_passed and success
    
    # Test searching for conditions/symptoms
    test_conditions = ["cough", "dental", "fracture", "surgery"]
    
    for condition in test_conditions:
        print(f"\n🔍 Testing text search for condition '{condition}'")
        success, results = search_by_text(client, condition)
        print_result(f"condition '{condition}'", success, "text search")
        all_passed = all_passed and success
    
    return all_passed

def test_pet_photo_search(client):
    """Test searching by pet photos"""
    print_header("PET PHOTO SEARCH")
    
    # Map each pet to their photo file - UPDATED to match actual files
    pet_photos = {
        "Emma": "emma_brooke-cagle-Ntm4C2lCWxQ-unsplash.jpg",
        "Jack": "jack_victor-g-N04FIfHhv_k-unsplash.jpg",
        "Max": "max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg",
        "Otis": "otis_karsten-winegeart-oU6KZTXhuvk-unsplash.jpg",
        "Rocky": "rocky_pauline-loroy-U3aF7hgUSrk-unsplash.jpg"
    }
    
    all_passed = True
    
    for pet_name, photo_file in pet_photos.items():
        photo_path = PET_PHOTOS_DIR / photo_file
        
        if not os.path.exists(photo_path):
            print(f"❌ Photo file not found: {photo_path}")
            all_passed = False
            continue
            
        print(f"\n🔍 Testing photo search with {pet_name}'s photo: {photo_file}")
        
        # Get embedding for the photo
        embedding = get_embedding(text="", image_path=photo_path)
        if embedding:
            # Search using the embedding
            success, _ = search_by_vector(client, embedding, pet_name)
            print_result(pet_name, success, "pet photo")
            all_passed = all_passed and success
        else:
            print(f"❌ Failed to get embedding for {pet_name}'s photo")
            all_passed = False
    
    return all_passed

def test_xray_search(client):
    """Test searching by X-ray images"""
    print_header("X-RAY SEARCH")
    
    # Map each pet to their X-ray file
    xray_files = {
        "Emma": "emma_1.2.410.200028.100.3.20190924.1430080534.21770.1.1.png",
        "Jack": "jack_1.2.410.200028.100.3.20190927.1023440667.730.1.1.png",
        "Max": "max_1.2.410.200028.100.3.20190925.1056080334.32610.1.2.png",
        "Otis": "otis_1.2.410.200028.100.3.20190926.1008510164.10580.1.2.png",
        "Rocky": "rocky_1.2.410.200028.100.3.20190927.1032580692.2539.1.1.png"
    }
    
    all_passed = True
    
    for pet_name, xray_file in xray_files.items():
        xray_path = XRAYS_DIR / xray_file
        
        if not os.path.exists(xray_path):
            print(f"❌ X-ray file not found: {xray_path}")
            all_passed = False
            continue
            
        print(f"\n🔍 Testing X-ray search with {pet_name}'s X-ray: {xray_file}")
        
        # Get embedding for the X-ray with a related search term
        embedding = get_embedding(text=f"X-ray for {pet_name}", image_path=xray_path)
        if embedding:
            # Search using the embedding
            success, _ = search_by_vector(client, embedding, pet_name)
            print_result(pet_name, success, "X-ray image")
            all_passed = all_passed and success
        else:
            print(f"❌ Failed to get embedding for {pet_name}'s X-ray")
            all_passed = False
    
    return all_passed

def test_combined_search(client):
    """Test searching with both text and images"""
    print_header("COMBINED TEXT + IMAGE SEARCH")
    
    # UPDATED test cases with correct filenames
    test_cases = [
        ("Emma", "dental", "emma_brooke-cagle-Ntm4C2lCWxQ-unsplash.jpg"),
        ("Jack", "emergency", "jack_victor-g-N04FIfHhv_k-unsplash.jpg"),
        ("Otis", "cough", "otis_1.2.410.200028.100.3.20190926.1008510164.10580.1.2.png"),
        ("Rocky", "broken leg", "rocky_1.2.410.200028.100.3.20190927.1032580692.2539.1.1.png")
    ]
    
    all_passed = True
    
    for pet_name, keyword, image_file in test_cases:
        if image_file.endswith(".jpg"):
            image_path = PET_PHOTOS_DIR / image_file
        else:
            image_path = XRAYS_DIR / image_file
            
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            all_passed = False
            continue
            
        print(f"\n🔍 Testing combined search for {pet_name} with keyword '{keyword}' and image {image_file}")
        
        # Get embedding for the combination
        embedding = get_embedding(text=keyword, image_path=image_path)
        if embedding:
            # Search using the embedding
            success, _ = search_by_vector(client, embedding, pet_name)
            print_result(pet_name, success, f"combined text+image ({keyword})")
            all_passed = all_passed and success
        else:
            print(f"❌ Failed to get embedding for combined search")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests and summarize results"""
    print("\n🧪 MULTIMODAL SEARCH TEST SCRIPT 🧪\n")
    
    # Initialize test summary
    test_results = {}
    
    # Check API status
    if not check_api_status():
        print("❌ Embedding API is not running. Please start it with `python app.py`")
        sys.exit(1)
        
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        print("❌ Failed to connect to Weaviate. Please make sure it's running.")
        sys.exit(1)
        
    # Check if database has content
    claim_count = get_claim_count(client)
    if claim_count == 0:
        print("❌ No claims found in the database. Please run the ingestion script first.")
        sys.exit(1)
    
    # Run all tests
    test_results["Text Search"] = test_text_search(client)
    test_results["Pet Photo Search"] = test_pet_photo_search(client)
    test_results["X-ray Search"] = test_xray_search(client)
    test_results["Combined Search"] = test_combined_search(client)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results.items():
        result = "✅ PASS" if passed else "❌ FAIL"
        print(f"{result}: {test_name}")
        all_passed = all_passed and passed
        
    print("\nOVERALL RESULT:", "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    
    # Return success/failure
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 