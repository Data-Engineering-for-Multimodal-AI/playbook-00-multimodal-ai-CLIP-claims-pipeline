# Multimodal Claims Processing Pipeline

A comprehensive end-to-end system for insurance claims processing using multimodal AI. This system enables insurance claim processors to search across historical claims using text, images (X-rays), and structured data for efficient claim assessment and processing.

## 🌟 Key Features

- **Multimodal Search**: Combine text descriptions and medical images to find similar historical claims
- **Vector Similarity**: Utilize CLIP embeddings for semantic understanding across modalities
- **Structured Data Integration**: Automatically link to policy details and claim history
- **Interactive UI**: Streamlit interface for intuitive claim search and review
- **Modular Architecture**: Separate components for embedding generation, vector storage, and UI

## 📊 Architecture Overview

![Architecture Diagram](images/architecture.png)

The pipeline consists of:

1. **FastAPI Embedding Service**: Generates embeddings from text and images using CLIP
2. **Weaviate Vector Database**: Stores and indexes claim embeddings for similarity search
3. **Streamlit Web Interface**: Provides an intuitive search and review experience
4. **Structured Data Store**: CSV files with claim history and policy information

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-claims-pipeline.git
cd multimodal-claims-pipeline

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the pipeline
./run_pipeline.sh
```

After running the setup script, you can access:
- **Streamlit UI**: http://localhost:8501
- **Embedding API**: http://localhost:8001
- **Weaviate Console**: http://localhost:8080

## 📋 Detailed Component Guide

### 1. CLIP Embedding API (`app.py`)

The embedding service is built with FastAPI and provides endpoints to convert text descriptions and images into vector embeddings using OpenAI's CLIP model.

Key features:
- Joint embeddings for text and image inputs
- Standardized vector dimensions (512)
- Error handling for malformed requests

### 2. Vector Database (Weaviate)

The Weaviate instance stores claim documents with:
- Embeddings from the CLIP model
- Metadata from structured sources
- Links to original images and text descriptions

### 3. Data Ingestion Pipeline

The `store_with_embedding.py` script:
- Processes sample claim data
- Retrieves embeddings from the API
- Stores documents in Weaviate
- Links with structured data

### 4. Search Interface (Streamlit)

The `streamlit_app.py` provides:
- Text-based search
- Image upload for similarity search
- Combined text+image search
- Display of matching claims with metadata

## 🔍 Testing

Run the automated tests to verify your setup:

```bash
python simple_test_app.py          # Test basic functionality
python test_claims_search.py --text "cough"  # Test text search
```

## 📚 Understanding the Data Model

The system uses the following data structure:

- **Claim Documents**: Core entity with embeddings and metadata
- **Policy Details**: Structured information about insurance coverage
- **Claim History**: Timeline of treatments and reimbursements

## 🔧 Troubleshooting

Common issues and solutions:

1. **422 Errors from API**: Ensure requests include the required `description` field
2. **No Results in Search**: Check that data ingestion completed successfully
3. **Docker Connection Issues**: Verify Weaviate container is running (`docker ps`)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 