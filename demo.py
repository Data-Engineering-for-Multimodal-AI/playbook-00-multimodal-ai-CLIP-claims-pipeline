import streamlit as st
import requests
import pandas as pd
import weaviate
from PIL import Image
import json
from pathlib import Path
import os
import time
import numpy as np

# Set up page config
st.set_page_config(
    page_title="Multimodal Claims Pipeline",
    page_icon="📋",
    layout="wide",
)

# Define the path to the data directory
DATA_DIR = Path("data")
XRAYS_DIR = DATA_DIR / "PetXrays"
SAMPLE_XRAYS = list(XRAYS_DIR.glob("*.png")) if XRAYS_DIR.exists() else []

# Initialize Weaviate client
@st.cache_resource
def get_weaviate_client():
    try:
        client = weaviate.Client("http://localhost:8080")
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

# Check Weaviate contents
def check_weaviate_contents(client):
    try:
        result = client.query.aggregate("ClaimDocument").with_meta_count().do()
        count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
        return count
    except Exception as e:
        st.error(f"Error checking Weaviate contents: {e}")
        return 0

# Get embedding from the API
def get_embedding(text, image_path=None, api_url="http://localhost:8001/embed"):
    try:
        if image_path and os.path.exists(image_path):
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
            st.error(f"Error getting embedding: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception when getting embedding: {e}")
        return None
    finally:
        # Close file handles
        for f in files.values():
            if hasattr(f, 'close'):
                f.close()

# Search for similar claims
def search_similar_claims(client, query_text=None, query_image=None, limit=5):
    try:
        # Get embedding for the query
        if query_text or query_image:
            embedding = get_embedding(query_text, query_image)
            if not embedding:
                st.error("Failed to get embedding for query")
                return []
        else:
            st.warning("Please provide either text or image query")
            return []
        
        # Vector search
        vector_results = []
        try:
            # Search with the vector
            vector_query = client.query.get(
                "ClaimDocument", 
                ["claim_id", "pet_name", "description", "image_path", "policy_id", 
                 "breed", "clinic", "deductible", "co_pay_percent", 
                 "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
            ).with_near_vector({
                "vector": embedding,
                "certainty": 0.6  # Lower certainty for more results
            }).with_limit(limit).do()
            
            if "data" in vector_query and "Get" in vector_query["data"] and "ClaimDocument" in vector_query["data"]["Get"]:
                vector_results = vector_query["data"]["Get"]["ClaimDocument"]
                
            # Add search method information
            for result in vector_results:
                result["search_method"] = "vector"
                result["score"] = 0.95  # Placeholder score for UI display
                
        except Exception as e:
            st.error(f"Vector search error: {e}")
            
        # Text search as backup or additional results
        text_results = []
        if query_text and len(vector_results) < limit:
            try:
                # Use fuzzy text search with like operator
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
                        },
                        {
                            "path": ["clinic"],
                            "operator": "like",
                            "valueText": f"*{query_text}*"
                        }
                    ]
                }
                
                text_query = client.query.get(
                    "ClaimDocument", 
                    ["claim_id", "pet_name", "description", "image_path", "policy_id", 
                     "breed", "clinic", "deductible", "co_pay_percent", 
                     "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
                ).with_where(where_filter).with_limit(limit).do()
                
                if "data" in text_query and "Get" in text_query["data"] and "ClaimDocument" in text_query["data"]["Get"]:
                    text_results = text_query["data"]["Get"]["ClaimDocument"]
                    
                # Add search method information
                for result in text_results:
                    result["search_method"] = "text"
                    result["score"] = 0.85  # Slightly lower score for text matches
                    
            except Exception as e:
                st.error(f"Text search error: {e}")
        
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
        st.error(f"Error in search: {e}")
        return []

# Get sample search terms
def get_sample_search_terms():
    terms = ["cough", "limp", "fracture", "xray", "infection"]
    if os.path.exists(DATA_DIR / "ClaimNotes"):
        claim_files = list((DATA_DIR / "ClaimNotes").glob("*.txt"))
        for claim_file in claim_files[:5]:  # Limit to first 5 to avoid too many
            with open(claim_file, "r") as f:
                description = f.read().strip()
                words = description.split()
                for word in words:
                    if len(word) > 4 and word.lower() not in terms:
                        terms.append(word.lower())
    return list(set(terms[:10]))  # Return at most 10 unique terms

# Parse claim history from JSON strings
def parse_claim_history(claim_history_json):
    history = []
    for item in claim_history_json:
        try:
            history.append(json.loads(item))
        except:
            continue
    return history

# Title and description
st.title("Pet Insurance Claim Search")
st.markdown("""
Search for similar insurance claims using text descriptions or pet X-ray images. 
The system will find the most relevant matches from our database.
""")

# Initialize Weaviate client
client = get_weaviate_client()

# System status
if client:
    claim_count = check_weaviate_contents(client)
    st.success(f"Connected to database with {claim_count} claims")
else:
    st.error("Failed to connect to Weaviate database")
    st.stop()

# Create tabs for different search methods
tab1, tab2 = st.tabs(["Text Search", "Image Search"])

with tab1:
    st.header("Search by Description")
    
    # Sample search terms
    sample_terms = get_sample_search_terms()
    if sample_terms:
        st.write("Try searching for these terms:")
        cols = st.columns(5)
        for i, term in enumerate(sample_terms):
            if cols[i % 5].button(term, key=f"term_{i}"):
                st.session_state.search_text = term
    
    # Text search input
    search_text = st.text_input(
        "Enter symptoms, condition, or pet details:", 
        key="search_text",
        placeholder="e.g., cough, limping, broken leg, labrador"
    )
    
    num_results = st.slider("Number of results:", 1, 10, 3, key="num_results_text")
    
    if st.button("Search", key="search_button_text"):
        if search_text:
            with st.spinner("Searching for similar claims..."):
                results = search_similar_claims(client, query_text=search_text, limit=num_results)
            
            if results:
                st.success(f"Found {len(results)} relevant claims")
                
                for i, result in enumerate(results):
                    with st.expander(f"#{i+1}: {result['pet_name']} - {result['description'][:50]}...", expanded=i==0):
                        cols = st.columns(2)
                        
                        with cols[0]:
                            st.subheader("Claim Details")
                            st.write(f"**Claim ID:** {result['claim_id']}")
                            st.write(f"**Pet Name:** {result['pet_name']}")
                            st.write(f"**Breed:** {result['breed']}")
                            st.write(f"**Clinic:** {result['clinic']}")
                            st.write(f"**Description:**")
                            st.write(result['description'])
                            
                            # Policy details
                            st.subheader("Policy Information")
                            st.write(f"**Policy ID:** {result['policy_id']}")
                            st.write(f"**Deductible:** ${result['deductible']}")
                            st.write(f"**Co-pay:** {result['co_pay_percent']}%")
                            
                            coverage = []
                            if result.get('xray_covered'):
                                coverage.append("X-ray")
                            if result.get('surgery_covered'):
                                coverage.append("Surgery")
                            if result.get('dental_covered'):
                                coverage.append("Dental")
                            
                            st.write(f"**Coverage:** {', '.join(coverage) if coverage else 'None'}")
                        
                        with cols[1]:
                            # Show image if available
                            image_path = result['image_path']
                            if image_path:
                                # Handle both relative and absolute paths
                                full_path = image_path
                                if not os.path.isabs(image_path):
                                    full_path = os.path.join(DATA_DIR, image_path)
                                
                                if os.path.exists(full_path):
                                    st.image(full_path, caption=f"{result['pet_name']}'s X-ray", width=300)
                                else:
                                    st.warning(f"Image not found: {image_path}")
                            
                            # Show claim history
                            st.subheader("Claim History")
                            history = parse_claim_history(result.get('claim_history', []))
                            
                            if history:
                                history_df = pd.DataFrame(history)
                                st.dataframe(history_df)
                            else:
                                st.info("No claim history available")
            else:
                st.warning("No matching claims found. Try different search terms.")
        else:
            st.warning("Please enter a search term")

with tab2:
    st.header("Search by X-ray Image")
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an X-ray image:", type=["png", "jpg", "jpeg"])
    
    # Optional text description
    image_description = st.text_input(
        "Optional description to refine the search:", 
        placeholder="e.g., dog, leg, fracture"
    )
    
    # Sample images
    if SAMPLE_XRAYS:
        st.write("Or select a sample X-ray:")
        sample_cols = st.columns(3)
        selected_sample = None
        
        for i, xray_path in enumerate(SAMPLE_XRAYS[:6]):  # Show up to 6 samples
            with sample_cols[i % 3]:
                if st.button(f"Sample {i+1}", key=f"sample_{i}"):
                    selected_sample = str(xray_path)
                    st.image(selected_sample, caption=f"Sample {i+1}", width=200)
    
    num_results = st.slider("Number of results:", 1, 10, 3, key="num_results_img")
    
    if st.button("Search", key="search_button_img"):
        # Determine which image to use
        image_path = None
        if uploaded_file:
            # Save uploaded image to temporary file
            temp_image = Image.open(uploaded_file)
            temp_path = "temp_uploaded_image.jpg"
            temp_image.save(temp_path)
            image_path = temp_path
            st.image(image_path, caption="Uploaded image", width=300)
        elif selected_sample:
            image_path = selected_sample
            st.image(image_path, caption="Selected sample", width=300)
        
        if image_path:
            with st.spinner("Searching for similar claims..."):
                results = search_similar_claims(client, query_text=image_description, query_image=image_path, limit=num_results)
            
            if results:
                st.success(f"Found {len(results)} relevant claims")
                
                for i, result in enumerate(results):
                    with st.expander(f"#{i+1}: {result['pet_name']} - {result['description'][:50]}...", expanded=i==0):
                        cols = st.columns(2)
                        
                        with cols[0]:
                            st.subheader("Claim Details")
                            st.write(f"**Claim ID:** {result['claim_id']}")
                            st.write(f"**Pet Name:** {result['pet_name']}")
                            st.write(f"**Breed:** {result['breed']}")
                            st.write(f"**Clinic:** {result['clinic']}")
                            st.write(f"**Description:**")
                            st.write(result['description'])
                            
                            # Policy details
                            st.subheader("Policy Information")
                            st.write(f"**Policy ID:** {result['policy_id']}")
                            st.write(f"**Deductible:** ${result['deductible']}")
                            st.write(f"**Co-pay:** {result['co_pay_percent']}%")
                            
                            coverage = []
                            if result.get('xray_covered'):
                                coverage.append("X-ray")
                            if result.get('surgery_covered'):
                                coverage.append("Surgery")
                            if result.get('dental_covered'):
                                coverage.append("Dental")
                            
                            st.write(f"**Coverage:** {', '.join(coverage) if coverage else 'None'}")
                        
                        with cols[1]:
                            # Show image if available
                            result_image_path = result['image_path']
                            if result_image_path:
                                # Handle both relative and absolute paths
                                full_path = result_image_path
                                if not os.path.isabs(result_image_path):
                                    full_path = os.path.join(DATA_DIR, result_image_path)
                                
                                if os.path.exists(full_path):
                                    st.image(full_path, caption=f"{result['pet_name']}'s X-ray", width=300)
                                else:
                                    st.warning(f"Image not found: {result_image_path}")
                            
                            # Show claim history
                            st.subheader("Claim History")
                            history = parse_claim_history(result.get('claim_history', []))
                            
                            if history:
                                history_df = pd.DataFrame(history)
                                st.dataframe(history_df)
                            else:
                                st.info("No claim history available")
            else:
                st.warning("No matching claims found. Try uploading a different image.")
        else:
            st.warning("Please upload an image or select a sample")

# Add a debug/admin section at the bottom
with st.expander("Admin Tools", expanded=False):
    st.subheader("Database Management")
    
    if st.button("Reingest Claims Data"):
        with st.spinner("Reingesting claims data..."):
            try:
                os.system("python store_with_embedding.py")
                st.success("Claims data reingested successfully!")
                time.sleep(2)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error reingesting claims data: {e}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Reset Weaviate Schema"):
            with st.spinner("Resetting Weaviate schema..."):
                try:
                    os.system("python weaviate_schema.py")
                    st.success("Weaviate schema reset successfully!")
                except Exception as e:
                    st.error(f"Error resetting Weaviate schema: {e}")
    
    with col2:
        if st.button("Check Claims Count"):
            count = check_weaviate_contents(client)
            st.info(f"Found {count} claims in the database") 