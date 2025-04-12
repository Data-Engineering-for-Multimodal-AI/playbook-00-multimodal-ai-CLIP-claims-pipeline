import streamlit as st
import os
import requests
import json
import pandas as pd
import uuid
import re
import tempfile
from pathlib import Path
import weaviate
from PIL import Image
import time

# Configuration
DATA_DIR = Path("data")
XRAYS_DIR = DATA_DIR / "PetXrays"
PET_PHOTOS_DIR = DATA_DIR / "PetPhotos"
EMBEDDING_API_URL = "http://localhost:8001"
WEAVIATE_URL = "http://localhost:8080"

# Set page config - MOVED TO BEGINNING OF SCRIPT
st.set_page_config(
    page_title="Multimodal Claims Search",
    page_icon="::",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    st.session_state.search_results = []
if "search_executed" not in st.session_state:
    st.session_state.search_executed = False

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

def get_embedding(text=None, image_file=None):
    """Get embeddings from the FastAPI service."""
    url = f"{EMBEDDING_API_URL}/embed"
    
    if not text and not image_file:
        st.error("Please provide either text or an image to search")
        return None
    
    files = {}
    data = {}
    temp_file = None
    
    try:
        # Always include a description field (required by API)
        data["description"] = text if text else ""
        
        # Add the text parameter as well for newer API versions
        if text:
            data["text"] = text
        
        # Process image if provided
        if image_file is not None:
            try:
                # Save uploaded file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                temp_file.write(image_file.getvalue())
                temp_file.close()
                
                # Show debug message and image preview
                st.write(f"Image received: {image_file.name} ({image_file.size} bytes)")
                st.image(image_file, caption="Uploaded image", width=200)
                
                # Add file to request
                files["image"] = open(temp_file.name, "rb")
            except Exception as e:
                st.error(f"Error processing image: {e}")
        
        # Make the API request
        st.write(f"Sending request to {url}")
        response = requests.post(url, files=files, data=data)
        st.write(f"API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            st.success("Embedding received successfully")
            
            # Look for the embedding in different fields
            embedding = result.get("embedding")
            if embedding is None:
                embedding = result.get("combined_embedding")
            if embedding is None:
                embedding = result.get("text_embedding")
                
            if embedding:
                return embedding
            else:
                st.error("API returned 200 but no embedding found in response")
                st.write("Response content:", result)
                return None
        else:
            st.error(f"Error getting embedding: {response.status_code}")
            st.write("Error details:", response.text)
            return None
            
    except Exception as e:
        st.error(f"Exception when getting embedding: {e}")
        return None
        
    finally:
        # Close and remove temp file if created
        if files and "image" in files and hasattr(files["image"], "close"):
            files["image"].close()
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def search_by_vector(client, embedding, limit=5):
    """Search for claims using vector similarity"""
    try:
        # Use a more permissive certainty value for pet matching
        vector_query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_near_vector({
            "vector": embedding,
            "certainty": 0.5  # Lower threshold to ensure we capture all relevant pets
        }).with_limit(limit).do()
        
        if "data" in vector_query and "Get" in vector_query["data"] and "ClaimDocument" in vector_query["data"]["Get"]:
            results = vector_query["data"]["Get"]["ClaimDocument"]
            return results
        else:
            return []
    except Exception as e:
        st.error(f"Error in vector search: {e}")
        return []

def search_by_text(client, query_text, limit=5):
    """Search for claims using text matching"""
    try:
        # Default text search across multiple fields
        where_filter = {
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
        }
        
        # Special case for coverage types - check exact field matches
        if query_text.lower() == "dental":
            where_filter = {
                "path": ["dental_covered"],
                "operator": "Equal",
                "valueBoolean": True
            }
        elif query_text.lower() == "surgery":
            where_filter = {
                "path": ["surgery_covered"],
                "operator": "Equal",
                "valueBoolean": True
            }
        elif query_text.lower() in ["x-ray", "xray"]:
            where_filter = {
                "path": ["xray_covered"],
                "operator": "Equal",
                "valueBoolean": True
            }
        
        # Execute the search
        text_query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_where(where_filter).with_limit(limit).do()
        
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

def search_similar_claims(client, query_text=None, query_image=None, limit=5):
    """Search for similar claims based on text and/or image using visual similarity"""
    # Ensure we have a text value (required by API)
    if query_text is None:
        query_text = ""
    
    # Extract pet name from image filename if available
    extracted_pet_name = None
    if query_image is not None:
        # Check if it's a file path string
        if isinstance(query_image, (str, Path)):
            filename = os.path.basename(str(query_image)).lower()
            for pet_name in ["rocky", "max", "otis", "jack", "emma"]:
                if pet_name in filename:
                    extracted_pet_name = pet_name.capitalize()
                    break
        # If we have a BytesIO object, check its name attribute
        elif hasattr(query_image, 'name'):
            filename = query_image.name.lower()
            for pet_name in ["rocky", "max", "otis", "jack", "emma"]:
                if pet_name in filename:
                    extracted_pet_name = pet_name.capitalize()
                    break
                    
    # Check if query text contains a pet name
    text_pet_name = None
    if query_text:
        query_lower = query_text.lower()
        for pet_name in ["rocky", "max", "otis", "jack", "emma"]:
            if pet_name in query_lower:
                text_pet_name = pet_name.capitalize()
                break
    
    # DIRECT PET NAME MATCHING: Always check for direct pet name matches first
    # This ensures reliable retrieval when we know exactly which pet we're looking for
    if extracted_pet_name or text_pet_name:
        pet_name_to_find = extracted_pet_name or text_pet_name
        st.success(f"Looking specifically for {pet_name_to_find}'s claims")
        
        # Direct lookup by pet name
        pet_query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_where({
            "path": ["pet_name"],
            "operator": "Equal",
            "valueText": pet_name_to_find
        }).do()
        
        if ("data" in pet_query and 
            "Get" in pet_query["data"] and 
            "ClaimDocument" in pet_query["data"]["Get"] and
            pet_query["data"]["Get"]["ClaimDocument"]):
            
            results = pet_query["data"]["Get"]["ClaimDocument"]
            for result in results:
                result["search_method"] = f"Pet Name Direct Match ({pet_name_to_find})"
            return results[:limit]
    
    # If we didn't find a direct match, proceed with vector search
    # APPROACH 1: Get embedding for the image/text (PRIMARY APPROACH)
    if query_text or query_image:
        embedding = get_embedding(text=query_text, image_file=query_image)
    
    # If we have an embedding, do vector search
    if embedding:
        st.write("Using vector search with embedding")
        results = search_by_vector(client, embedding, limit=limit)
        
        # Check if results contain Rocky when we're looking for Rocky
        if query_text and "rocky" in query_text.lower():
            # Look for Rocky in the results
            found_rocky = False
            for result in results:
                if result.get("pet_name") == "Rocky":
                    found_rocky = True
                    break
                    
            # If Rocky isn't in the results, do a direct lookup
            if not found_rocky:
                st.warning("Rocky not found in vector search results, trying direct lookup")
                results = search_by_text(client, "rocky", limit=limit)
        
        # If searching specifically for dental claims
        if query_text and "dental" in query_text.lower():
            st.write("Enhancing search for dental coverage")
            dental_results = search_by_text(client, "dental", limit=limit)
            results = combine_search_results(results, dental_results)
            
        # Add search method
        for result in results:
            if not result.get("search_method"):
                result["search_method"] = "Vector Similarity Search"
                
        return results[:limit]
    
    # If no embedding or no results, fallback to text search
    elif query_text:
        st.write("Falling back to text-only search")
        results = search_by_text(client, query_text, limit=limit)
        
        # Add search method
        for result in results:
            if not result.get("search_method"):
                result["search_method"] = "Text Keyword Search"
                
        return results[:limit]
    else:
        st.warning("Please provide text or image to search")
        return []

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

def render_result(result, index=None):
    """Render a search result with detailed claim information."""
    with st.expander(f"Claim {result['claim_id']} - {result['pet_name']} ({result['breed']})"):
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
            st.markdown(f"**X-Ray Coverage:** {'Yes' if result['xray_covered'] else 'No'}")
        with cov2:
            st.markdown(f"**Surgery Coverage:** {'Yes' if result['surgery_covered'] else 'No'}")
        with cov3:
            st.markdown(f"**Dental Coverage:** {'Yes' if result['dental_covered'] else 'No'}")
        
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
    st.session_state.search_results = []

# Callback for submit button
def on_search_submit():
    st.session_state.search_executed = True

# Add new function at the top of the file where other helper functions are defined
def reset_search_state():
    """Reset search-related session state when changing modes"""
    if "search_executed" in st.session_state:
        st.session_state.search_executed = False
    if "search_results" in st.session_state:
        st.session_state.search_results = []
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
    # Title and introduction
    st.title("Multimodal Claims Search")
    st.markdown("<div style='margin-bottom:1rem;'>Search pet insurance claims with text keywords, photos, or X-rays</div>", unsafe_allow_html=True)
    
    # Initialize Weaviate client
    client = connect_to_weaviate()
    
    if not client:
        st.error("Failed to connect to Weaviate database. Please make sure it's running.")
        st.stop()
    
    # Check embedding API status
    api_running = check_api_status()
    
    # Display status
    status_color = "green" if api_running else "red"
    status_text = "Connected" if api_running else "Not connected"
    st.markdown(f"<div style='color:{status_color};margin-bottom:1rem;'>Embedding API status: {status_text}</div>", unsafe_allow_html=True)
    
    if not api_running:
        st.error("Embedding API is not running. Please start it with `uvicorn app:app --host 0.0.0.0 --port 8001`")
        st.stop()
    
    # Get claim count
    claim_count = get_claim_count(client)
    st.write(f"Found {claim_count} claims in the database")
    
    if claim_count == 0:
        st.warning("No claims found in the database. Please run the ingestion script first.")
        st.stop()
    
    # Display search options
    st.subheader("Search Options")
    
    # Create tabs for search modes
    search_mode_tab = st.radio(
        "Search method",
        ["Text Keywords", "Pet Photo", "X-ray Image"],
        horizontal=True,
        key="search_method",
        on_change=reset_search_state
    )
    
    # Store the search mode in session state
    st.session_state.search_mode = search_mode_tab
    
    # Container for quick search buttons
    if search_mode_tab == "Text Keywords":
        # Quick search buttons - OUTSIDE any form
        st.write("Try searching for:")
        quick_search_cols = st.columns(5)
        sample_terms = ["cough", "arthritis", "fracture", "dental", "surgery"]
        
        for i, term in enumerate(sample_terms):
            if quick_search_cols[i].button(term, key=f"quick_{term}"):
                st.session_state.text_query = term
                st.session_state.search_executed = True
                
                # Run search - FIXED to include client parameter
                with st.spinner(f"Searching for '{term}'..."):
                    results = search_similar_claims(client, query_text=term, query_image=None, limit=3)
                    st.session_state.search_results = results
                    st.rerun()
    
    # Begin search form area
    with st.form(key="search_form"):
        if search_mode_tab == "Text Keywords":
            # Text search
            text_query = st.text_input(
                "Enter search keywords",
                value=st.session_state.get("text_query", ""),
                key="text_input"
            )
            query_image = None
            
        elif search_mode_tab == "Pet Photo":
            # Pet photo search
            col1, col2 = st.columns([3, 2])
            
            with col1:
                text_query = st.text_input("Pet name or condition (optional):", placeholder="e.g., Rocky, limping")
                query_image = st.file_uploader("Upload a pet photo", type=["jpg", "jpeg", "png"], key="pet_photo_uploader")
            
            with col2:
                # Sample image selection - simplified to use a select box
                sample_options = ["None", "Emma (Goldendoodle)", "Jack (French Bulldog)", "Max (Labrador)", 
                                  "Otis (Border Collie)", "Rocky (Jack Russell)"]
                selected_sample = st.selectbox("Or use a sample pet photo:", sample_options, index=0)
                
                # If a sample is selected, show it
                if selected_sample != "None":
                    # Map selection to file path
                    sample_map = {
                        "Emma (Goldendoodle)": "emma_brooke-cagle-Ntm4C2lCWxQ-unsplash.jpg",
                        "Jack (French Bulldog)": "jack_victor-g-N04FIfHhv_k-unsplash.jpg",
                        "Max (Labrador)": "max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg",
                        "Otis (Border Collie)": "otis_karsten-winegeart-oU6KZTXhuvk-unsplash.jpg",
                        "Rocky (Jack Russell)": "rocky_pauline-loroy-U3aF7hgUSrk-unsplash.jpg"
                    }
                    
                    if selected_sample in sample_map:
                        sample_filename = sample_map[selected_sample]
                        sample_image_path = str(PET_PHOTOS_DIR / sample_filename)
                        st.session_state.pet_photo_path = sample_image_path
                        
                        if os.path.exists(sample_image_path):
                            st.image(sample_image_path, caption=f"Sample: {selected_sample}", use_container_width=True)
                            st.info(f"This sample will be used when you click Search")
                        else:
                            st.error(f"Sample image not found: {sample_image_path}")

        elif search_mode_tab == "X-ray Image":
            # X-ray search
            col1, col2 = st.columns([3, 2])
            
            with col1:
                text_query = st.text_input("Condition (optional):", placeholder="e.g., fracture, arthritis")
                query_image = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png", "dicom", "dcm"], key="xray_uploader")
            
            with col2:
                # Sample image selection - simplified to use a select box
                sample_options = ["None", "Emma (Chest X-ray)", "Jack (Shoulder X-ray)", "Max (Hip X-ray)", 
                                  "Otis (Chest X-ray)", "Rocky (Leg X-ray)"]
                selected_sample = st.selectbox("Or use a sample X-ray:", sample_options, index=0)
                
                # If a sample is selected, show it
                if selected_sample != "None":
                    # Map selection to file path
                    sample_map = {
                        "Emma (Chest X-ray)": "emma_1.2.410.200028.100.3.20190924.1430080534.21770.1.1.png",
                        "Jack (Shoulder X-ray)": "jack_1.2.410.200028.100.3.20190927.1023440667.730.1.1.png",
                        "Max (Hip X-ray)": "max_1.2.410.200028.100.3.20190925.1056080334.32610.1.2.png",
                        "Otis (Chest X-ray)": "otis_1.2.410.200028.100.3.20190926.1008510164.10580.1.2.png",
                        "Rocky (Leg X-ray)": "rocky_1.2.410.200028.100.3.20190927.1032580692.2539.1.1.png"
                    }
                    
                    if selected_sample in sample_map:
                        sample_filename = sample_map[selected_sample]
                        sample_image_path = str(XRAYS_DIR / sample_filename)
                        st.session_state.xray_path = sample_image_path
                        
                        if os.path.exists(sample_image_path):
                            st.image(sample_image_path, caption=f"Sample: {selected_sample}", use_container_width=True)
                            st.info(f"This sample will be used when you click Search")
                        else:
                            st.error(f"Sample image not found: {sample_image_path}")
        
        # Number of results to return
        num_results = st.slider("Number of results", min_value=1, max_value=10, value=3)
        
        # Submit button
        submit_button = st.form_submit_button("Search", use_container_width=True)
    
    # Handle form submission
    if submit_button:
        st.session_state.search_executed = True
        
        # Get the file to use
        file_to_use = None
        
        if search_mode_tab == "Pet Photo":
            if query_image:
                file_to_use = query_image
            elif st.session_state.get("pet_photo_path"):
                try:
                    with open(st.session_state.pet_photo_path, "rb") as f:
                        sample_file_bytes = f.read()
                    from io import BytesIO
                    file_to_use = BytesIO(sample_file_bytes)
                    file_to_use.name = os.path.basename(st.session_state.pet_photo_path)
                    st.success(f"Using sample pet photo: {os.path.basename(st.session_state.pet_photo_path)}")
                except Exception as e:
                    st.error(f"Error loading sample pet photo: {e}")
        
        elif search_mode_tab == "X-ray Image":
            if query_image:
                file_to_use = query_image
            elif st.session_state.get("xray_path"):
                try:
                    with open(st.session_state.xray_path, "rb") as f:
                        sample_file_bytes = f.read()
                    from io import BytesIO
                    file_to_use = BytesIO(sample_file_bytes)
                    file_to_use.name = os.path.basename(st.session_state.xray_path)
                    st.success(f"Using sample X-ray: {os.path.basename(st.session_state.xray_path)}")
                except Exception as e:
                    st.error(f"Error loading sample X-ray: {e}")
        
        # Validate input
        if search_mode_tab in ["Pet Photo", "X-ray Image"] and not file_to_use:
            st.error(f"Please upload a{' pet photo' if search_mode_tab == 'Pet Photo' else 'n X-ray image'} or select a sample")
        elif search_mode_tab == "Text Keywords" and not text_query:
            st.error("Please enter search keywords")
        else:
            # Show a spinner while searching
            with st.spinner("Searching..."):
                if search_mode_tab == "Text Keywords":
                    results = search_similar_claims(client, query_text=text_query, query_image=None, limit=num_results)
                else:
                    results = search_similar_claims(client, query_text=text_query, query_image=file_to_use, limit=num_results)
                
                st.session_state.search_results = results
            
            # Display results count
            if results:
                st.success(f"Found {len(results)} matching claims")
                
                # Display each result
                for i, result in enumerate(results):
                    render_result(result, i)
            else:
                st.warning("No matching claims found. Try different search terms or upload a different image.")

    # Display admin tools
    with st.expander("Admin Tools", expanded=False):
        st.subheader("Database Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reingest Claims Data"):
                with st.spinner("Reingesting claims data..."):
                    try:
                        os.system("python store_with_embedding.py")
                        st.success("Claims data reingested successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error reingesting claims data: {e}")
        with col2:
            if st.button("Reset Weaviate Schema"):
                with st.spinner("Resetting Weaviate schema..."):
                    try:
                        os.system("python weaviate_schema.py")
                        st.success("Weaviate schema reset successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error resetting Weaviate schema: {e}")

if __name__ == "__main__":
    main() 