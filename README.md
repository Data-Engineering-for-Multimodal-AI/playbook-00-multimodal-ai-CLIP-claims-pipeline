# Multimodal Claims Pipeline

A multimodal insurance claims processing system that enables intelligent search across X-ray images, text descriptions, and structured policy data.

## Overview

This system allows insurance claim processors to:
- Search for similar claims using X-ray images
- Find relevant claims with text descriptions of symptoms
- Upload photos of pets to retrieve their claim history
- Access policy coverage information and claim history in one view

It combines machine learning for visual and text understanding with a vector database for fast similarity search.

![System Architecture](https://mermaid.ink/img/pako:eNp1kk1PwzAMhv9KlBMIaV3XD7QLJ4QEEkJwQBw4JK6JlqYlSVdtqPvvOFvXTnDYKbZfP3lt59SWKtCUdkH7mnPUxBdSo8G1EtYZ5MQ4zsDGcS3rPRpOBGzs_lmBKGvQp4ijB5-uQxw3SoPnZLOUw6eMr0A58hHwcllKKBbQ2KrXjRVEkEJ2YWEhFpBsL8fZRThNhBXxD4G8FaiwK9kRrZr4pEHxdVgrb1RjO-_z5fL6Znl3fxM23ouBXAQE1Yc05A5VEUKT0MYbWCCLfUe0jmkBx5BPx1cXC_p-6dMhzODj-XnFUCzgCcUAzRcMY4SoikVRTpvO4dXTLqzUL2Ij7d4o8-0S4DTnFDx5M1Dp9Xs-6zHfUZ6NNg3qjK8GKnpjTvITlLKBPG-lCpEfj36RyrUbcB2tRVoZe_iHZ7ShMZXYxpLmX5Rnp-lRLYLfJzWj-OTVUgXfYJw1oZ4_Hxl4HzYopLWEqw-Ru73tKWWwM6VwqtZUoBFvKY14gw5zU9JM6I7mLOc9xnOmIJ8vkqnbM-PJDzq9AhQ?type=png)

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Full Pipeline**
   ```bash
   ./run_pipeline.sh
   ```
   This will:
   - Start Weaviate (or connect to existing instance)
   - Set up the schema
   - Start the CLIP embedding API
   - Ingest sample data
   - Launch the Streamlit UI

3. **Access the Interface**
   - Open [http://localhost:8501](http://localhost:8501) in your browser

## System Components

### 1. Embedding API (app.py)
- FastAPI service that generates embeddings from text and images
- Uses CLIP model to create joint embedding space
- Runs on port 8001 by default

### 2. Vector Database (Weaviate)
- Stores claim documents with embeddings
- Enables semantic search across modalities
- Runs in Docker on port 8080

### 3. Search Interface (streamlit_app.py)
- Streamlit app for claim search and exploration
- Upload images, enter text, or combine both
- View claim history and policy details
- Runs on port 8501 by default

### 4. Data Ingestion (store_with_embedding.py)
- Processes claim notes, images and policy data
- Extracts embeddings and stores in Weaviate
- Manages structured data relationships

## Data Organization

The system expects data to be organized in the following structure:
```
data/
  PetPhotos/           # Pet photos for search
  PetXrays/            # X-ray images for search 
  ClaimNotes/          # Text descriptions of claims
  StructuredData/
    policy_details.csv # Policy information 
    claim_history.csv  # Claim history records
```

## Usage Guide

### Searching with Text
1. Enter a symptom, condition, or policy term in the search box
2. Click "Search" to find similar claims
3. Explore results showing policy details and history

### Searching with Images
1. Upload an X-ray image or pet photo
2. Optionally enter additional text
3. Click "Search" to find visually similar cases

### Special Case: Pet Search
The system has special handling for pet searches:
- Uploading a pet photo while typing its name prioritizes that pet's claims
- Rocky's claims are consistently found when his photo is uploaded

## Testing

Use the test script to validate search functionality:
```bash
# Test X-ray search
python test_claims_search.py --xray data/PetXrays/emma_1.2.410.200028.100.3.20190924.1430080534.21770.1.1.png

# Test pet photo search
python test_claims_search.py --photo data/PetPhotos/rocky_pauline-loroy-U3aF7hgUSrk-unsplash.jpg

# Test text search
python test_claims_search.py --text "dental"
```

## Troubleshooting

### Port Conflicts
- If port 8001 is in use: Edit the `EMBEDDING_API_PORT` in app.py
- If port 8501 is in use: Streamlit will automatically try 8502, 8503, etc.

### Embedding API Issues
- 422 Errors: Ensure your request includes either 'description' or 'image' parameter
- Connection Errors: Check that the embedding API is running at http://localhost:8001

### Streamlit Performance
For better Streamlit performance, install Watchdog:
```bash
xcode-select --install  # Mac only
pip install watchdog
```

### Weaviate Issues
- If schema reset fails: Run `python weaviate_schema.py` manually
- No results: Ensure data ingestion completed with `python store_with_embedding.py`

### Claim History Parsing
If claim history doesn't display correctly:
- Check that the JSON string format matches expected schema
- Enable debug toggle in the UI to view raw data format

## License

MIT License - see LICENSE file for details. 