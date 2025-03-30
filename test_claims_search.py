#!/usr/bin/env python
"""
Test script for multimodal claims search capabilities.
Tests three search scenarios:
1. Search by X-ray image
2. Search by pet photo
3. Search by text keyword (e.g., "cough", "arthritis")

This script directly uses the Weaviate database and embedding API
to verify that the search functionality works correctly without the Streamlit UI.
"""

import weaviate
import requests
import json
import os
from pathlib import Path
import random
import sys
import argparse
from PIL import Image
import time
import pprint

# Configuration
DATA_DIR = Path("data")
XRAYS_DIR = DATA_DIR / "PetXrays"
PET_PHOTOS_DIR = DATA_DIR / "PetPhotos"
EMBEDDING_API_URL = "http://localhost:8001"
WEAVIATE_URL = "http://localhost:8080"

# Pretty printer for nicely formatted output
pp = pprint.PrettyPrinter(indent=2)

def print_separator(title=""):
    """Print a separator line with optional title"""
    width = 80
    if title:
        print("\n" + "=" * 20 + f" {title} " + "=" * (width - len(title) - 22) + "\n")
    else:
        print("\n" + "=" * width + "\n")

def connect_to_weaviate():
    """Connect to Weaviate database"""
    try:
        client = weaviate.Client(WEAVIATE_URL)
        print("✅ Connected to Weaviate database")
        return client
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
        print(f"❌ Embedding API is not available: {e}")
        return False

def get_claim_count(client):
    """Get the number of claims in the database"""
    try:
        result = client.query.aggregate("ClaimDocument").with_meta_count().do()
        count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
        print(f"📊 Found {count} claims in the database")
        return count
    except Exception as e:
        print(f"❌ Error checking claim count: {e}")
        return 0

def get_embedding(text=None, image_path=None):
    """Get embedding from the API for text and/or image"""
    url = f"{EMBEDDING_API_URL}/embed"
    
    if not text and not image_path:
        print("Error: Please provide either text or an image path")
        return None
    
    files = {}
    data = {}
    
    if text:
        data["text"] = text
        # Add description field for backward compatibility
        data["description"] = text
    
    if image_path:
        if os.path.exists(image_path):
            files["image"] = open(image_path, "rb")
            print(f"Sending image: {image_path}")
        else:
            print(f"Error: Image file not found at {image_path}")
            return None
    
    try:
        print(f"Getting embedding from {url}")
        if files:
            response = requests.post(url, data=data, files=files)
        else:
            response = requests.post(url, json=data)
        
        print(f"API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Successfully received embedding")
            return result.get("embedding")
        else:
            print(f"Error getting embedding: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except Exception as e:
        print(f"Exception when getting embedding: {e}")
        return None
    
    finally:
        # Close file handles
        for f in files.values():
            if hasattr(f, 'close'):
                f.close()

def search_by_vector(embedding, limit=5):
    """Search by vector embedding"""
    client = connect_to_weaviate()
    if not client:
        return []
    
    try:
        query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_near_vector({
            "vector": embedding,
            "certainty": 0.7  # Adjust this threshold as needed
        }).with_limit(limit).do()
        
        if "data" in query and "Get" in query["data"] and "ClaimDocument" in query["data"]["Get"]:
            results = query["data"]["Get"]["ClaimDocument"]
            print(f"Found {len(results)} results with vector search")
            # Mark search method
            for result in results:
                result["search_method"] = "Vector Search"
            return results
        else:
            print("No vector search results found")
            return []
    
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

def search_by_text(query_text, limit=5, where_filter=None):
    """Search by text using Weaviate's where filter"""
    client = connect_to_weaviate()
    if not client:
        return []
    
    try:
        # Create the base query
        query_builder = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        )
        
        # Add a where filter if provided
        if where_filter:
            query_builder = query_builder.with_where(where_filter)
        else:
            # Default to searching claim_note
            query_builder = query_builder.with_where({
                "path": ["claim_note"],
                "operator": "Like",
                "valueText": f"*{query_text}*"
            })
        
        # Execute the query with a limit
        query = query_builder.with_limit(limit).do()
        
        if "data" in query and "Get" in query["data"] and "ClaimDocument" in query["data"]["Get"]:
            results = query["data"]["Get"]["ClaimDocument"]
            # Mark search method
            for result in results:
                result["search_method"] = "Text Search"
            return results
        else:
            print("No text search results found")
            return []
    
    except Exception as e:
        print(f"Error in text search: {e}")
        return []

def combine_search_results(vector_results, text_results, limit=5):
    """Combine vector and text search results, removing duplicates"""
    seen_claim_ids = set()
    combined_results = []
    
    # Add vector results first (they're typically more relevant)
    for result in vector_results:
        if result["claim_id"] not in seen_claim_ids:
            seen_claim_ids.add(result["claim_id"])
            combined_results.append(result)
    
    # Add text results that aren't duplicates
    for result in text_results:
        if result["claim_id"] not in seen_claim_ids:
            seen_claim_ids.add(result["claim_id"])
            combined_results.append(result)
            
            # Stop if we have enough results
            if len(combined_results) >= limit:
                break
    
    return combined_results[:limit]  # Return only up to limit results

def display_claim(claim, index=None):
    """Display claim information in a readable format"""
    prefix = f"RESULT #{index + 1}: " if index is not None else ""
    print(f"\n{prefix}CLAIM ID: {claim['claim_id']}")
    print(f"Pet Name: {claim['pet_name']}")
    print(f"Breed: {claim['breed']}")
    print(f"Clinic: {claim['clinic']}")
    print(f"Description: {claim['claim_note'][:100]}..." if len(claim['claim_note']) > 100 else f"Description: {claim['claim_note']}")
    
    print("\nPolicy Details:")
    print(f"  Policy ID: {claim['policy_id']}")
    print(f"  Deductible: ${claim['deductible']}")
    print(f"  Co-pay: {claim['co_pay_percent']}%")
    
    coverage = []
    if claim.get('xray_covered'):
        coverage.append("X-ray")
    if claim.get('surgery_covered'):
        coverage.append("Surgery")
    if claim.get('dental_covered'):
        coverage.append("Dental")
    
    print(f"  Coverage: {', '.join(coverage) if coverage else 'None'}")
    
    # Image path
    if claim.get('xray_image'):
        print(f"\nImage: {claim['xray_image']}")
    
    # Claim history
    if claim.get('claim_history'):
        print("\nClaim History:")
        for i, history_item in enumerate(claim['claim_history']):
            try:
                history_data = json.loads(history_item)
                print(f"  {i+1}. Date: {history_data.get('date', 'N/A')}, Type: {history_data.get('type', 'N/A')}, Amount: ${history_data.get('amount', 'N/A')}")
            except:
                pass

def search_similar_claims(query_text=None, query_image=None, limit=3):
    """Searches for similar claims based on text and/or image"""
    embedding = None
    search_method = "None"
    
    # Get embedding for search
    if query_text or query_image:
        embedding = get_embedding(text=query_text, image_path=query_image)
    
    # If we have an embedding, do vector search
    if embedding:
        print("Using vector search with embedding")
        search_method = "Vector"
        results = search_by_vector(embedding, limit=limit)
        
        if results:
            # If searching specifically for dental claims
            if query_text and "dental" in query_text.lower():
                print("Enhancing search for dental coverage")
                search_method = "Dental + Vector"
                dental_results = search_by_text("dental", limit=limit, 
                                               where_filter={
                                                   "path": ["dental_covered"],
                                                   "operator": "Equal",
                                                   "valueBoolean": True
                                               })
                results = combine_search_results(results, dental_results)
            
            # Mark all results with the search method
            for result in results:
                result["search_method"] = search_method
            
            return results
    
    # If no embedding or no results, fallback to text search
    if query_text:
        print("Falling back to text-only search")
        search_method = "Text-only"
        results = search_by_text(query_text, limit=limit)
        
        # Mark all results with the search method
        for result in results:
            result["search_method"] = search_method
        
        return results
    else:
        print("Please provide text or image to search")
        return []

def test_xray_search(client, xray_path=None):
    """Test searching by X-ray image"""
    print_separator("TEST 1: SEARCH BY X-RAY IMAGE")
    
    # Find a random X-ray if not provided
    if not xray_path and XRAYS_DIR.exists():
        xray_files = list(XRAYS_DIR.glob("*.png"))
        if xray_files:
            xray_path = str(random.choice(xray_files))
    
    if not xray_path or not os.path.exists(xray_path):
        print(f"❌ No X-ray image found to test with. Looked in {XRAYS_DIR}")
        return False
    
    print(f"🔍 Testing search with X-ray image: {xray_path}")
    
    # Get the pet name from the image filename if possible
    pet_name = os.path.basename(xray_path).split('_')[0] if '_' in os.path.basename(xray_path) else None
    description = f"X-ray for {pet_name}" if pet_name else ""
    
    # Search for similar claims
    results = search_similar_claims(query_text=description, query_image=xray_path, limit=3)
    
    # Display results
    if results:
        print("\n📋 Search Results:")
        for i, result in enumerate(results):
            display_claim(result, i)
        return True
    else:
        print("❌ No matching claims found for the X-ray image")
        return False

def test_pet_photo_search(client, photo_path=None):
    """Test searching by pet photo"""
    print_separator("TEST 2: SEARCH BY PET PHOTO")
    
    # Find a random pet photo if not provided
    if not photo_path and PET_PHOTOS_DIR.exists():
        photo_files = list(PET_PHOTOS_DIR.glob("*.jpg")) + list(PET_PHOTOS_DIR.glob("*.png"))
        if photo_files:
            photo_path = str(random.choice(photo_files))
    
    if not photo_path or not os.path.exists(photo_path):
        print(f"❌ No pet photo found to test with. Looked in {PET_PHOTOS_DIR}")
        return False
    
    print(f"🔍 Testing search with pet photo: {photo_path}")
    
    # Get the pet name from the image filename if possible
    pet_name = os.path.basename(photo_path).split('_')[0] if '_' in os.path.basename(photo_path) else None
    description = pet_name if pet_name else ""
    
    # Search for similar claims
    results = search_similar_claims(query_text=description, query_image=photo_path, limit=3)
    
    # Display results
    if results:
        print("\n📋 Search Results:")
        for i, result in enumerate(results):
            display_claim(result, i)
        return True
    else:
        print("❌ No matching claims found for the pet photo")
        return False

def test_text_search(client, query_text=None):
    """Test searching by text keyword"""
    print_separator("TEST 3: SEARCH BY TEXT KEYWORD")
    
    # Use a default query if none provided
    if not query_text:
        test_queries = ["cough", "arthritis", "fracture", "surgery", "x-ray"]
        query_text = random.choice(test_queries)
    
    print(f"🔍 Testing search with text query: '{query_text}'")
    
    # Search for similar claims
    results = search_similar_claims(query_text=query_text, limit=3)
    
    # Display results
    if results:
        print("\n📋 Search Results:")
        for i, result in enumerate(results):
            display_claim(result, i)
        return True
    else:
        print(f"❌ No matching claims found for '{query_text}'")
        return False

def get_available_resources():
    """Check and report available resources for testing"""
    print_separator("CHECKING AVAILABLE RESOURCES")
    
    resources = {
        "xrays": [],
        "photos": [],
        "claim_notes": [],
        "structured_data": {
            "policy_details": None,
            "claim_history": None
        }
    }
    
    # Check for X-rays
    if XRAYS_DIR.exists():
        resources["xrays"] = list(XRAYS_DIR.glob("*.png"))
        print(f"✅ Found {len(resources['xrays'])} X-ray images in {XRAYS_DIR}")
    else:
        print(f"❌ X-ray directory not found: {XRAYS_DIR}")
    
    # Check for pet photos
    if PET_PHOTOS_DIR.exists():
        resources["photos"] = list(PET_PHOTOS_DIR.glob("*.jpg")) + list(PET_PHOTOS_DIR.glob("*.png"))
        print(f"✅ Found {len(resources['photos'])} pet photos in {PET_PHOTOS_DIR}")
    else:
        print(f"❌ Pet photos directory not found: {PET_PHOTOS_DIR}")
    
    # Check for claim notes
    claim_notes_dir = DATA_DIR / "ClaimNotes"
    if claim_notes_dir.exists():
        resources["claim_notes"] = list(claim_notes_dir.glob("*.txt"))
        print(f"✅ Found {len(resources['claim_notes'])} claim notes in {claim_notes_dir}")
    else:
        print(f"❌ Claim notes directory not found: {claim_notes_dir}")
    
    # Check for structured data
    structured_data_dir = DATA_DIR / "StructuredData"
    policy_file = structured_data_dir / "policy_details.csv"
    claims_file = structured_data_dir / "claim_history.csv"
    
    if policy_file.exists():
        resources["structured_data"]["policy_details"] = policy_file
        print(f"✅ Found policy details file: {policy_file}")
    else:
        print(f"❌ Policy details file not found: {policy_file}")
    
    if claims_file.exists():
        resources["structured_data"]["claim_history"] = claims_file
        print(f"✅ Found claim history file: {claims_file}")
    else:
        print(f"❌ Claim history file not found: {claims_file}")
    
    return resources

def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description="Test multimodal claims search capabilities")
    parser.add_argument("--xray", type=str, help="Path to a specific X-ray image to test with")
    parser.add_argument("--photo", type=str, help="Path to a specific pet photo to test with")
    parser.add_argument("--text", type=str, help="Specific text query to test with")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--test1", action="store_true", help="Run test 1: X-ray search")
    parser.add_argument("--test2", action="store_true", help="Run test 2: Pet photo search")
    parser.add_argument("--test3", action="store_true", help="Run test 3: Text search")
    
    args = parser.parse_args()
    
    print("\n🔍 MULTIMODAL CLAIMS SEARCH TEST SCRIPT 🔍\n")
    
    # Check embedding API status
    if not check_api_status():
        print("\n❌ Embedding API is not available. Make sure it's running at http://localhost:8001")
        print("   You can start it with: uvicorn app:app --host 0.0.0.0 --port 8001")
        return
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        print("\n❌ Failed to connect to Weaviate. Make sure it's running at http://localhost:8080")
        return
    
    # Check claim count
    claim_count = get_claim_count(client)
    if claim_count == 0:
        print("\n❌ No claims found in the database. Please run the ingestion script first.")
        print("   You can run it with: python store_with_embedding.py")
        return
    
    # Check available resources
    resources = get_available_resources()
    
    # Determine which tests to run
    run_test1 = args.all or args.test1
    run_test2 = args.all or args.test2
    run_test3 = args.all or args.test3
    
    # If no specific tests selected, run all
    if not (run_test1 or run_test2 or run_test3):
        run_test1 = run_test2 = run_test3 = True
    
    # Run the selected tests
    if run_test1:
        test_xray_search(client, args.xray)
    
    if run_test2:
        test_pet_photo_search(client, args.photo)
    
    if run_test3:
        test_text_search(client, args.text)
    
    print_separator("TEST SUMMARY")
    print("All tests completed. Check the results above to see if they match expected outcomes.")

if __name__ == "__main__":
    main() 