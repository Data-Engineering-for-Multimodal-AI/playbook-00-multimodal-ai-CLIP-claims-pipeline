import weaviate
import requests
import argparse
from pathlib import Path
import json
from PIL import Image
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple

# API endpoint for embeddings
EMBEDDING_API_URL = "http://localhost:8001/embed"
FILE_EMBEDDING_API_URL = "http://localhost:8001/embed_file"

def get_embeddings_from_api(description, image_path=None):
    """Get embeddings from the API."""
    if image_path:
        # If we have a file path
        data = {"description": description, "image_path": image_path}
        response = requests.post(FILE_EMBEDDING_API_URL, data=data)
    else:
        # If we don't have an image
        data = {"description": description}
        response = requests.post(FILE_EMBEDDING_API_URL, data=data)
        
    if response.status_code != 200:
        print(f"Error getting embeddings: {response.text}")
        return None
        
    return response.json()

def check_weaviate_contents(client):
    """Check how many claims are in the Weaviate database"""
    try:
        result = client.query.aggregate("ClaimDocument").with_meta_count().do()
        count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
        print(f"Found {count} claims in Weaviate")
        return count
    except Exception as e:
        print(f"Error checking Weaviate contents: {e}")
        return 0

def get_all_claims(client):
    """Get all claims from Weaviate"""
    try:
        result = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "description", "image_path"]
        ).with_limit(100).do()
        
        return result["data"]["Get"]["ClaimDocument"]
    except Exception as e:
        print(f"Error retrieving claims: {e}")
        return []

def text_similarity(query, claim_description):
    """Calculate simple text similarity score based on keyword matching."""
    # Convert to lowercase
    query = query.lower()
    description = claim_description.lower()
    
    # Simple keyword matching
    query_terms = query.split()
    
    # Count matching terms
    matching_terms = sum(1 for term in query_terms if term in description)
    
    # Normalize by query length
    if len(query_terms) > 0:
        return matching_terms / len(query_terms)
    return 0

def search_similar_claims(client, query_text=None, query_image=None, embedding_api_url="http://localhost:8001/embed", limit=5):
    """Search for claims similar to the provided text and/or image"""
    try:
        # Get embedding for the query
        if query_text or query_image:
            embedding = get_embedding(query_text, query_image, embedding_api_url)
            if not embedding:
                print("Failed to get embedding for query")
                return []
        else:
            print("Please provide either text or image query")
            return []
        
        # Vector search
        vector_results = []
        try:
            # Search with the vector
            vector_query = client.query.get(
                "ClaimDocument", 
                ["claim_id", "pet_name", "description", "image_path", "policy_id", 
                 "breed", "clinic", "deductible", "co_pay_percent", 
                 "xray_covered", "surgery_covered", "dental_covered"]
            ).with_near_vector({
                "vector": embedding,
                "certainty": 0.7
            }).with_limit(limit).do()
            
            if "data" in vector_query and "Get" in vector_query["data"] and "ClaimDocument" in vector_query["data"]["Get"]:
                vector_results = vector_query["data"]["Get"]["ClaimDocument"]
                
            # Add search method information
            for result in vector_results:
                result["search_method"] = "vector"
                
        except Exception as e:
            print(f"Vector search error: {e}")
            
        # Text search as backup or additional results
        text_results = []
        if query_text and len(vector_results) < limit:
            try:
                # Use BM25 search
                where_filter = {
                    "operator": "or",
                    "operands": [
                        {
                            "path": ["description"],
                            "operator": "like",
                            "valueText": f"*{query_text}*"
                        },
                        {
                            "path": ["pet_name"],
                            "operator": "like",
                            "valueText": f"*{query_text}*"
                        },
                        {
                            "path": ["breed"],
                            "operator": "like",
                            "valueText": f"*{query_text}*"
                        }
                    ]
                }
                
                text_query = client.query.get(
                    "ClaimDocument", 
                    ["claim_id", "pet_name", "description", "image_path", "policy_id", 
                     "breed", "clinic", "deductible", "co_pay_percent", 
                     "xray_covered", "surgery_covered", "dental_covered"]
                ).with_where(where_filter).with_limit(limit).do()
                
                if "data" in text_query and "Get" in text_query["data"] and "ClaimDocument" in text_query["data"]["Get"]:
                    text_results = text_query["data"]["Get"]["ClaimDocument"]
                    
                # Add search method information
                for result in text_results:
                    result["search_method"] = "text"
                    
            except Exception as e:
                print(f"Text search error: {e}")
        
        # Combine results, removing duplicates
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
        
    except Exception as e:
        print(f"Error in search: {e}")
        return []

def get_embedding(text, image_path=None, api_url="http://localhost:8001/embed"):
    """Get embedding from the API"""
    try:
        if image_path and Path(image_path).exists():
            # Both text and image query
            files = {"image": open(image_path, "rb")}
            data = {"text": text if text else ""}
        else:
            # Text-only query
            files = {}
            data = {"text": text}
            
        response = requests.post(api_url, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()["embedding"]
        else:
            print(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        print(f"Exception when getting embedding: {e}")
        return None
    finally:
        # Close file handles
        for f in files.values():
            if hasattr(f, 'close'):
                f.close()

def format_results(results):
    """Format search results for display"""
    formatted = []
    
    for idx, result in enumerate(results, 1):
        item = {
            "rank": idx,
            "claim_id": result["claim_id"],
            "pet_name": result["pet_name"],
            "description": result.get("description", "")[:100] + "..." if len(result.get("description", "")) > 100 else result.get("description", ""),
            "search_method": result.get("search_method", "unknown")
        }
        formatted.append(item)
    
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Search for similar insurance claims")
    parser.add_argument("--text", type=str, help="Text query")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    args = parser.parse_args()
    
    # Initialize Weaviate client
    client = weaviate.Client("http://localhost:8080")
    
    # Check if we have claims in the database
    count = check_weaviate_contents(client)
    if count == 0:
        print("No claims found in Weaviate. Please run the ingestion script first.")
        return
    
    # If no search arguments provided, list all claims
    if not args.text and not args.image:
        claims = get_all_claims(client)
        print(f"Listing all claims ({len(claims)}):")
        for claim in claims:
            print(f"Claim ID: {claim['claim_id']}, Pet: {claim['pet_name']}")
        return
    
    # Search for similar claims
    results = search_similar_claims(
        client, 
        query_text=args.text, 
        query_image=args.image,
        limit=args.limit
    )
    
    # Format and display results
    if results:
        formatted = format_results(results)
        print("\nSearch results:")
        for item in formatted:
            print(f"#{item['rank']} - Claim {item['claim_id']} - {item['pet_name']} - " +
                  f"Method: {item['search_method']}")
            print(f"  {item['description']}")
            print()
    else:
        print("No matching claims found.")

if __name__ == "__main__":
    main() 