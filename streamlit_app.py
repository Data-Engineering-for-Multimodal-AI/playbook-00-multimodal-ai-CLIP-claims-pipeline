import streamlit as st
import weaviate
import requests
import json
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import time
import uuid
import re
import tempfile

# Configuration
DATA_DIR = Path("data")
XRAYS_DIR = DATA_DIR / "PetXrays"
PET_PHOTOS_DIR = DATA_DIR / "PetPhotos"
EMBEDDING_API_URL = "http://localhost:8001/embed"
WEAVIATE_URL = "http://localhost:8080"

# Initialize session state for storing selections
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "Text Keywords"
if "use_pet_sample" not in st.session_state:
    st.session_state.use_pet_sample = False
if "use_xray_sample" not in st.session_state:
    st.session_state.use_xray_sample = False
if "text_query" not in st.session_state:
    st.session_state.text_query = ""
if "pet_photo_path" not in st.session_state:
    st.session_state.pet_photo_path = ""
if "xray_path" not in st.session_state:
    st.session_state.xray_path = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_executed" not in st.session_state:
    st.session_state.search_executed = False

# Page configuration
st.set_page_config(
    page_title="🐾 PetInsure Claims Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for improved appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E63B5;
        margin-bottom: 1rem;
    }
    .search-results {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 1rem;
    }
    .data-info {
        color: #777;
        font-size: 0.8rem;
        margin-bottom: 1rem;
    }
    .policy-details {
        background-color: #f0f7ff;
        padding: 0.8rem;
        border-radius: 5px;
        margin-bottom: 0.8rem;
    }
    .claim-history {
        background-color: #fff9f0;
        padding: 0.8rem;
        border-radius: 5px;
    }
    .sample-btn {
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        cursor: pointer;
        font-size: 0.8rem;
    }
    .search-method {
        background-color: #1E63B5;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        display: inline-block;
    }
    .stRadio > div {
        display: flex;
        flex-direction: row;
    }
    .stRadio label {
        margin-right: 1rem;
    }
    .stImage > img {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    .search-btn {
        background-color: #1E63B5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        font-weight: 500;
        cursor: pointer;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_resource
def connect_to_weaviate():
    """Connect to Weaviate database"""
    try:
        client = weaviate.Client(WEAVIATE_URL)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

def check_api_status():
    """Check if the embedding API is running"""
    try:
        response = requests.get("http://localhost:8001/")
        return response.status_code == 200
    except Exception as e:
        return False

def get_claim_count(client):
    """Get the number of claims in the database"""
    try:
        result = client.query.aggregate("ClaimDocument").with_meta_count().do()
        count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
        return count
    except Exception as e:
        return 0

def get_embedding(text=None, image=None):
    """Get embeddings from the FastAPI service."""
    url = f"{EMBEDDING_API_URL}/embed"
    
    if not text and not image:
        st.error("Please provide either text or an image to search")
        return None
    
    files = {}
    data = {}
    
    if text:
        data["text"] = text
    
    temp_file = None
    try:
        if image is not None:
            # Log the image file info for debugging
            st.write(f"Processing image: {image.name}, size: {image.size}, type: {image.type}")
            
            # Save the uploaded image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(image.getvalue())
            temp_file.close()
            
            # Add the image file to the request
            files["image"] = (image.name, open(temp_file.name, "rb"), image.type)
    
        # Log the API request for debugging
        st.write(f"Sending request to {url} with data: {data} and files: {list(files.keys())}")
        
        # Make the API request
        if files:
            response = requests.post(url, data=data, files=files)
        else:
            response = requests.post(url, json=data)
        
        # Log the API response status for debugging
        st.write(f"API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            st.success("Embedding successfully received")
            return result.get("embedding")
        else:
            st.error(f"Failed to get embedding: {response.text}")
            return None
    
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        # Close any open file handles
        for f in files.values():
            if hasattr(f[1], 'close'):
                f[1].close()

def search_by_vector(embedding, limit=5):
    """Search for claims using vector similarity"""
    client = connect_to_weaviate()
    if not client:
        st.error("Failed to connect to Weaviate")
        return []
        
    try:
        # Use a more permissive certainty value for pet matching
        vector_query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_near_vector({
            "vector": embedding,
            "certainty": 0.6  # Lower threshold to catch more matches
        }).with_limit(limit).do()
        
        if "data" in vector_query and "Get" in vector_query["data"] and "ClaimDocument" in vector_query["data"]["Get"]:
            results = vector_query["data"]["Get"]["ClaimDocument"]
            return results
        else:
            return []
    except Exception as e:
        st.error(f"Error in vector search: {e}")
        return []

def search_by_text(query_text, limit=5, where_filter=None):
    """Search for claims using text matching"""
    client = connect_to_weaviate()
    if not client:
        st.error("Failed to connect to Weaviate")
        return []
        
    try:
        # Create base query
        query_builder = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        )
        
        # Use provided filter or create a default one
        if where_filter:
            query_builder = query_builder.with_where(where_filter)
        else:
            # Default text search across multiple fields
            query_builder = query_builder.with_where({
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
                    },
                    {
                        "path": ["breed"],
                        "operator": "Like",
                        "valueText": f"*{query_text}*"
                    },
                    {
                        "path": ["clinic"],
                        "operator": "Like",
                        "valueText": f"*{query_text}*"
                    }
                ]
            })
        
            # Special case for coverage types - check exact field matches
            if query_text.lower() == "dental":
                query_builder = query_builder.with_where({
                    "path": ["dental_covered"],
                    "operator": "Equal",
                    "valueBoolean": True
                })
            elif query_text.lower() == "surgery":
                query_builder = query_builder.with_where({
                    "path": ["surgery_covered"],
                    "operator": "Equal",
                    "valueBoolean": True
                })
            elif query_text.lower() in ["x-ray", "xray"]:
                query_builder = query_builder.with_where({
                    "path": ["xray_covered"],
                    "operator": "Equal",
                    "valueBoolean": True
                })
        
        # Execute the query
        text_query = query_builder.with_limit(limit).do()
        
        if "data" in text_query and "Get" in text_query["data"] and "ClaimDocument" in text_query["data"]["Get"]:
            results = text_query["data"]["Get"]["ClaimDocument"]
            return results
        else:
            return []
    except Exception as e:
        st.error(f"Error in text search: {e}")
        return []

def combine_search_results(vector_results, text_results, limit=5):
    """Combine vector and text search results, removing duplicates"""
    seen_claim_ids = set()
    combined_results = []
    
    # Add vector results first (they're typically more relevant)
    for result in vector_results:
        if result.get("claim_id") and result["claim_id"] not in seen_claim_ids:
            seen_claim_ids.add(result["claim_id"])
            combined_results.append(result)
    
    # Add text results that aren't duplicates
    for result in text_results:
        if result.get("claim_id") and result["claim_id"] not in seen_claim_ids:
            seen_claim_ids.add(result["claim_id"])
            combined_results.append(result)
            
            # Stop if we have enough results
            if len(combined_results) >= limit:
                break
    
    return combined_results[:limit]  # Return only up to limit results

def search_similar_claims(query_text=None, query_image=None, limit=3):
    """Search for similar claims based on text and/or image."""
    embedding = None
    search_method = "None"
    
    # Get embedding for search
    if query_text or query_image:
        embedding = get_embedding(text=query_text, image=query_image)
    
    # If we have an embedding, do vector search
    if embedding:
        st.write("Using vector search with embedding")
        search_method = "Vector"
        results = search_by_vector(embedding, limit=limit)
        
        # If searching specifically for dental claims
        if query_text and "dental" in query_text.lower():
            st.write("Enhancing search for dental coverage")
            search_method = "Dental + Vector"
            dental_results = search_by_text("dental", limit=limit, 
                                            where_filter={
                                                "path": ["dental_covered"],
                                                "operator": "Equal",
                                                "valueBoolean": True
                                            })
            results = combine_search_results(results, dental_results)
    
    # If no embedding or no results, fallback to text search
    elif query_text:
        st.write("Falling back to text-only search")
        search_method = "Text-only"
        results = search_by_text(query_text, limit=limit)
    else:
        st.warning("Please provide text or image to search")
        return [], "No search performed"
    
    # Log search results
    st.write(f"Search complete. Method: {search_method}. Found {len(results)} results.")
    
    return results, search_method

def parse_claim_history(claim_history_json):
    """Parse claim history from JSON string or list of strings"""
    history_items = []
    
    # Handle empty or None values
    if not claim_history_json:
        return []
    
    try:
        # If we have a string that looks like a JSON array, parse it
        if isinstance(claim_history_json, str):
            if claim_history_json.strip().startswith('['):
                # This is a JSON array string, parse it directly
                parsed_items = json.loads(claim_history_json)
                if isinstance(parsed_items, list):
                    return parsed_items
            elif claim_history_json.strip().startswith('{'):
                # This is a single JSON object string
                parsed_item = json.loads(claim_history_json)
                if isinstance(parsed_item, dict):
                    return [parsed_item]
        
        # If we have a list, process each item
        if isinstance(claim_history_json, list):
            for item in claim_history_json:
                # Handle if the item is already a dict
                if isinstance(item, dict):
                    history_items.append(item)
                # Handle if the item is a JSON string
                elif isinstance(item, str):
                    try:
                        parsed_item = json.loads(item)
                        if isinstance(parsed_item, dict):
                            history_items.append(parsed_item)
                        elif isinstance(parsed_item, list):
                            history_items.extend(parsed_item)
                    except json.JSONDecodeError:
                        # If not JSON, try to extract information using regex
                        st.warning(f"Could not parse history item as JSON: {item[:50]}...")
        
        return history_items
    except Exception as e:
        st.warning(f"Error parsing claim history: {str(e)[:100]}")
        return []

def render_result(result):
    """Render a search result with detailed claim information."""
    with st.expander(f"📋 Claim {result['claim_id']} - {result['pet_name']} ({result['breed']})"):
        st.markdown(f"**Search Method:** {result.get('search_method', 'Unknown')}")
        
        # Basic claim info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Pet:** {result['pet_name']}")
            st.markdown(f"**Breed:** {result['breed']}")
            st.markdown(f"**Clinic:** {result['clinic']}")
        with col2:
            st.markdown(f"**Policy ID:** {result['policy_id']}")
            st.markdown(f"**Deductible:** ${result['deductible']}")
            st.markdown(f"**Co-Pay:** {result['co_pay_percent']}%")
        
        # Coverage info
        st.markdown("### Coverage")
        cov1, cov2, cov3 = st.columns(3)
        with cov1:
            st.markdown(f"**X-Ray Coverage:** {'✅' if result['xray_covered'] else '❌'}")
        with cov2:
            st.markdown(f"**Surgery Coverage:** {'✅' if result['surgery_covered'] else '❌'}")
        with cov3:
            st.markdown(f"**Dental Coverage:** {'✅' if result['dental_covered'] else '❌'}")
        
        # Claim description
        st.markdown("### Claim Description")
        st.markdown(result['claim_note'])
        
        # Claim history
        st.markdown("### Claim History")
        claim_history = result.get('claim_history')
        
        # Add a debug toggle
        debug = st.checkbox(f"Debug raw claim history for {result['claim_id']}", value=False)
        if debug:
            st.write("Raw claim history data:", claim_history)
        
        try:
            if claim_history:
                if isinstance(claim_history, str):
                    # Try to parse JSON string
                    try:
                        history_items = json.loads(claim_history)
                        if isinstance(history_items, list):
                            df = pd.DataFrame(history_items)
                            st.dataframe(df)
                        else:
                            st.write("Claim history is not in expected list format")
                            st.write(history_items)
                    except json.JSONDecodeError:
                        st.error(f"Error parsing claim history JSON: {claim_history[:50]}...")
                elif isinstance(claim_history, list):
                    # Already a list
                    df = pd.DataFrame(claim_history)
                    st.dataframe(df)
                else:
                    st.error(f"Unexpected claim history format: {type(claim_history)}")
            else:
                st.info("No claim history available")
        except Exception as e:
            st.error(f"Error processing claim history: {e}")
        
        # X-ray image if available
        if result.get('xray_image'):
            st.markdown("### X-Ray Image")
            st.image(result['xray_image'], caption=f"X-Ray for {result['pet_name']}")

# Callback for text sample buttons
def set_text_sample(term):
    st.session_state.text_query = term
    
# Callback for search modes
def on_search_mode_change():
    # Reset some state variables on mode change
    st.session_state.search_executed = False
    st.session_state.search_results = None

# Callback for submit button
def on_search_submit():
    st.session_state.search_executed = True

# Add new function at the top of the file where other helper functions are defined
def reset_search_state():
    """Reset search-related session state when changing modes"""
    if "search_executed" in st.session_state:
        st.session_state.search_executed = False
    if "search_results" in st.session_state:
        st.session_state.search_results = None
    if "sample_pet_displayed" in st.session_state:
        st.session_state.sample_pet_displayed = False
    if "sample_xray_displayed" in st.session_state:
        st.session_state.sample_xray_displayed = False

# Add the missing run_search function
def run_search(client):
    """Run search with the current session state"""
    search_mode = st.session_state.search_mode
    text_query = st.session_state.text_query
    
    # Create file_to_use based on session state
    file_to_use = None
    if search_mode == "Pet Photo" and st.session_state.get("pet_photo_path"):
        with open(st.session_state.pet_photo_path, "rb") as f:
            sample_file_bytes = f.read()
        from io import BytesIO
        file_to_use = BytesIO(sample_file_bytes)
        file_to_use.name = os.path.basename(st.session_state.pet_photo_path)
    elif search_mode == "X-ray Image" and st.session_state.get("xray_path"):
        with open(st.session_state.xray_path, "rb") as f:
            sample_file_bytes = f.read()
        from io import BytesIO
        file_to_use = BytesIO(sample_file_bytes)
        file_to_use.name = os.path.basename(st.session_state.xray_path)
    
    # Perform search based on mode
    if search_mode == "Text Keywords":
        results = search_similar_claims(client, query_text=text_query, query_image=None, limit=3)
    else:
        results = search_similar_claims(client, query_text=text_query, query_image=file_to_use, limit=3)
    
    # Store results in session state
    st.session_state.search_results = results
    
    # Force a rerun to display results
    st.rerun()

# Main application
def main():
    # The page config is now set at the top of the file
    st.title("🐾 PetInsure Claims Search")
    
    # Initialize Weaviate client
    client = connect_to_weaviate()
    if not client:
        st.error("Failed to connect to Weaviate")
        return
    
    # Display API status
    api_status = check_api_status()
    if api_status:
        st.success("✅ Embedding API is running")
    else:
        st.error("❌ Embedding API is not available")
    
    # Get current claim count
    claim_count = get_claim_count(client)
    st.write(f"📊 Database contains {claim_count} claims")
    
    # Search form
    with st.form("search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            query_text = st.text_input("Search by keywords:", placeholder="Enter search terms...")
        
        with col2:
            query_image = st.file_uploader("Upload X-ray or pet photo:", type=["jpg", "jpeg", "png"])
        
        num_results = st.slider("Maximum number of results:", min_value=1, max_value=10, value=3)
        submitted = st.form_submit_button("🔍 Search")
    
    # Process search when form is submitted
    if submitted:
        with st.spinner("Searching for matching claims..."):
            if not query_text and query_image is None:
                st.warning("Please enter search text or upload an image")
            else:
                # Search for similar claims
                results, search_method = search_similar_claims(
                    query_text=query_text, 
                    query_image=query_image, 
                    limit=num_results
                )
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} matching claims using {search_method} search")
                    for result in results:
                        render_result(result)
                else:
                    st.warning("No matching claims found. Try different search terms or upload a different image.")

if __name__ == "__main__":
    main() 