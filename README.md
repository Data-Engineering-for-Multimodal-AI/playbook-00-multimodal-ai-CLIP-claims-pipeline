# Multimodal Claims Pipeline

A multimodal insurance claims processing system that enables intelligent search across X-ray images, text descriptions, and structured policy data for pet insurance.

## Overview

This system allows insurance claim processors to:
- Search for similar claims using X-ray images
- Find relevant claims with text descriptions of symptoms
- Upload photos of pets to retrieve their claim history
- Access policy coverage information and claim history in one view

The pipeline uses a CLIP (Contrastive Language-Image Pretraining) model to create embeddings for both text and images, placing them in the same vector space. This enables powerful multimodal search that understands both visual and textual content.

## Technical Architecture

The system consists of four main components:

1. **Embedding API (app.py)**
   - FastAPI service that generates embeddings from text and images
   - Uses OpenAI's CLIP model to create joint embedding space
   - Processes both text descriptions and pet images/X-rays

2. **Vector Database (Weaviate)**
   - Stores claim documents with embeddings
   - Enables semantic search across text and images
   - Maintains relationships between claims, policies, and history

3. **Data Ingestion Pipeline (store_with_embedding.py)**
   - Processes claim notes, X-ray images, and policy data
   - Extracts embeddings and stores in Weaviate
   - Links claims with their policy details and history

4. **Search Interface (streamlit_app.py)**
   - User-friendly interface for multimodal search
   - Supports uploading images, entering text, or both
   - Displays search results with policy details and claim history

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **Docker** - Required for running Weaviate
- **Git** - For cloning the repository

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb1-mmclaimspipeline.git
cd pb1-mmclaimspipeline
```

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Start Weaviate (Vector Database)

Weaviate runs as a Docker container. The script will start it, but you can also run it manually:

```bash
docker run -d --name weaviate -p 8080:8080 \
    --env QUERY_DEFAULTS_LIMIT=25 \
    --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
    --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    --env DEFAULT_VECTORIZER_MODULE=none \
    --restart unless-stopped \
    semitechnologies/weaviate:1.23.9
```

You can check if Weaviate is running by visiting http://localhost:8080/v1/.well-known/ready in your browser. It should return `{"status":"OK"}`.

### 4. Run the Pipeline

The easiest way to start all components is with the provided script:

```bash
# Make sure the script is executable
chmod +x run_pipeline.sh

# Run the pipeline
./run_pipeline.sh
```

This script will:
1. Start Weaviate if not already running
2. Set up the Weaviate schema
3. Start the CLIP embedding API
4. Ingest the sample claim data
5. Launch the Streamlit UI

### 5. Access the User Interface

Once the pipeline is running, open http://localhost:8501 in your browser to access the Streamlit interface.

## Expected Output

### When Running run_pipeline.sh

```
=== Starting Multimodal Claims Pipeline ===
Checking for required Python packages...
Dependencies look good. Continuing setup...

Checking if Weaviate is running...
SUCCESS: Weaviate is already running.

Setting up Weaviate schema...
Existing schema deleted
Created schema with ClaimDocument class
Weaviate schema setup completed.

Starting the CLIP embedding API...
INFO:     Started server process [28842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
SUCCESS: Embedding API started successfully!

Ingesting claims data into Weaviate...
Weaviate is operational
Existing schema deleted
Created schema with ClaimDocument class
Processing claims in data/ClaimNotes...
Processing claim CLM001...
Processing claim CLM002...
Processing claim CLM003...
Processing claim CLM004...
Processing claim CLM005...
SUCCESS: Claims data ingested successfully!

Checking data in Weaviate...
Found 5 claims in Weaviate

Starting the Streamlit demo app...
Access the app at http://localhost:8501

Press Ctrl+C to stop the pipeline
```

### When Starting the Streamlit App

The Streamlit interface will show:
- A search interface with options for text, image, or combined search
- Sample buttons for quick demonstrations
- A clean results view showing matching claims with policy information

## Using the Application

### Text-Based Search

1. Select "Text Keywords" search mode
2. Type a symptom or condition like "limp", "dental", or "fracture"
3. Click "Search" to find relevant claims
4. Results will show matching claims with their:
   - Pet name and breed
   - Claim description
   - Policy details (coverage, deductibles)
   - Claim history

**Example Output:**

```
Searching for: "dental issues"

Results:
1. Claim ID: CLM001
   Pet: Emma (Goldendoodle)
   Description: Emma has been experiencing dental issues with plaque buildup and bad breath. Requires dental cleaning and possible extraction.
   Similarity Score: 0.89
   
   Policy Details:
   - Policy ID: POL-1001
   - Deductible: $100
   - Co-pay: 15%
   - Dental Coverage: Yes (Covered)
```

### Image-Based Search (X-rays)

1. Select "X-ray Image" search mode
2. Upload an X-ray image or use one of the sample images
3. Optionally add text to refine the search
4. Click "Search"
5. Results will show claims with similar X-ray images

**Example Output when searching with an X-ray image:**

```
Searching with X-ray image

Results:
1. Claim ID: CLM003
   Pet: Max (Labrador Retriever) 
   X-ray Image: [thumbnail shown]
   Description: Max has a visible fracture in right front leg from jumping off the deck. Need surgery consultation.
   Similarity Score: 0.78
   
   Policy Details:
   - Policy ID: POL-1003
   - Surgery Coverage: Yes (Covered)
```

### Pet Photo Search

1. Select "Pet Photo" search mode  
2. Upload a photo of a pet or use a sample photo
3. Click "Search"
4. Results will prioritize claims for the specific pet

**Example Output when searching with Rocky's photo:**

```
Searching with pet photo

Results:
1. Claim ID: CLM005
   Pet: Rocky (Jack Russell Terrier)
   Description: Rocky has been limping and showing pain in his back right leg after playing at the dog park.
   Policy Details:
   - Policy ID: POL-1005
   - X-ray Coverage: Yes (Covered)
   - Surgery Coverage: Yes (Covered)
```

## Testing and Verification

The repository includes test scripts to verify functionality:

```bash
# Test X-ray search
python test_claims_search.py --xray data/PetXrays/emma_1.2.410.200028.100.3.20190924.1430080534.21770.1.1.png

# Test pet photo search
python test_claims_search.py --photo data/PetPhotos/rocky_pauline-loroy-U3aF7hgUSrk-unsplash.jpg

# Test text search
python test_claims_search.py --text "dental"

# Run all tests
python run_tests.py
```

## Data Structure

The system uses the following data structure:

```
data/
  ClaimNotes/                # Text descriptions of claims (format: CLM001_description.txt)
  PetXrays/                  # X-ray images for search (format: petname_dicom.png)
  PetPhotos/                 # Pet photos for search (format: petname_photo.jpg)
  StructuredData/
    policy_details.csv       # Policy information
    claim_history.csv        # Claim history records
```

## Troubleshooting

### Port Conflicts
- If port 8001 is in use: Edit the `EMBEDDING_API_PORT` in run_pipeline.sh
- If port 8501 is in use: Streamlit will automatically try the next available port (8502, etc.)

### Embedding API Issues
- If you see 422 Errors: Ensure your request includes either 'description' or 'image' parameter
- If you get Connection Errors: Check that the embedding API is running at http://localhost:8001

### Weaviate Issues
- If Weaviate isn't starting: Check Docker is running with `docker ps`
- If schema reset fails: Run `python weaviate_schema.py` manually
- If no results appear: Ensure data ingestion completed with `python store_with_embedding.py`

### Memory Issues
- If you encounter memory errors with CLIP model loading:
  - Close memory-intensive applications
  - Decrease batch size in app.py if needed

## Extending the System

### Adding New Data

To add new claims:

1. Add claim note files in `data/ClaimNotes/` with format `CLM[ID]_description.txt`
2. Add X-ray images in `data/PetXrays/` with pet name prefix
3. Add pet photos in `data/PetPhotos/` with pet name prefix
4. Update `data/StructuredData/policy_details.csv` with policy information
5. Update `data/StructuredData/claim_history.csv` with claim history
6. Run data ingestion: `python store_with_embedding.py`

### Modifying the Schema

To change the data structure:

1. Edit the schema definition in `weaviate_schema.py`
2. Update the data processing in `store_with_embedding.py`
3. Update the UI in `streamlit_app.py` to reflect new fields

## Project Details

### Dependencies
- FastAPI - For the embedding API service
- Streamlit - For the user interface
- Weaviate - Vector database for semantic search
- Transformers/CLIP - For multimodal embeddings
- PyTorch - Deep learning framework
- Pandas - Data processing

### Resources

- Weaviate Documentation: https://weaviate.io/developers/weaviate/
- CLIP Model: https://github.com/openai/CLIP
- Streamlit Documentation: https://docs.streamlit.io/

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
