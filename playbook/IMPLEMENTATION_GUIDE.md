# Multimodal Claims Pipeline: Implementation Guide for Data Engineers

This guide walks through building a production-ready multimodal claims processing system from scratch. As a data engineer, you'll learn how to integrate vector embeddings, structured data, and a search interface into a cohesive system.

## System Architecture Overview

![Architecture Diagram](images/architecture.png)

The system follows a modern, microservices-based architecture with:

1. **Vector Embedding Service**: A FastAPI application that generates embeddings from text and images
2. **Vector Database**: Weaviate for storing and querying vector embeddings
3. **Structured Data Store**: CSV files with claim history and policy details
4. **Search UI**: Streamlit interface for end users

The data flow follows these steps:
- Text and images are converted to embeddings via the FastAPI service
- Embeddings are stored in Weaviate alongside metadata
- Users search through the Streamlit UI
- Results combine vector similarity with structured data

## Implementation Roadmap

This guide follows a systematic approach:

1. **Environment Setup**: Preparing your development environment
2. **Data Organization**: Structuring sample data
3. **Vector Store Setup**: Configuring Weaviate
4. **Embedding API**: Building the FastAPI service with CLIP
5. **Data Ingestion**: Loading data into Weaviate
6. **Search Interface**: Building the Streamlit app
7. **Testing & Optimization**: Validating performance

## 1. Environment Setup

### Prerequisites
- Python 3.10+ (required for compatibility with the CLIP model)
- Docker (for running Weaviate)
- 8GB+ RAM (for model inference)

### Initialize Project

```bash
# Create project directory
mkdir multimodal-claims-pipeline
cd multimodal-claims-pipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Create requirements.txt
```

Create a `requirements.txt` file with these dependencies:

```
fastapi==0.104.0
uvicorn==0.23.2
transformers==4.34.1
pillow==10.0.1
torch==2.0.1
streamlit==1.28.0
weaviate-client==3.26.7
pandas==2.1.1
python-multipart==0.0.6
pydantic==2.4.2
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 2. Data Organization

Create a structured data directory:

```bash
mkdir -p data/{PetPhotos,PetXrays,ClaimNotes,StructuredData}
```

### Sample Data Creation

For a real implementation, you'd connect to your data warehouse. For this guide, we'll use sample data:

#### 2.1 Create Pet X-ray Images

Place X-ray images in `data/PetXrays/`:
- Naming convention: `{pet_name}_{dicom_id}.png`
- Example: `max_1.2.410.200028.100.3.20190927.1023440667.730.1.1.png`

#### 2.2 Create Pet Photos

Place pet photos in `data/PetPhotos/`:
- Naming convention: `{pet_name}_{source}.jpg`
- Example: `rocky_pauline-loroy-U3aF7hgUSrk-unsplash.jpg`

#### 2.3 Create Claim Notes

Create text files in `data/ClaimNotes/`:
- Naming convention: `{claim_id}.txt`
- Example: `CLM001.txt` with content describing the claim

#### 2.4 Create Structured Data CSVs

Create `data/StructuredData/policy_details.csv`:

```csv
policy_id,pet_name,breed,birthdate,policy_start_date,deductible,co_pay_percent,dental_covered,xray_covered,surgery_covered
POL-1001,Emma,Goldendoodle,2018-03-15,2022-01-01,100,15,TRUE,TRUE,TRUE
POL-1002,Jack,French Bulldog,2019-07-22,2022-02-15,110,20,FALSE,TRUE,TRUE
POL-1003,Max,Labrador Retriever,2017-11-30,2022-03-01,120,25,FALSE,TRUE,TRUE
POL-1004,Otis,Border Collie,2020-01-05,2022-04-10,130,30,TRUE,TRUE,FALSE
POL-1005,Rocky,Golden Retriever,2019-04-18,2022-05-20,140,35,FALSE,FALSE,TRUE
```

Create `data/StructuredData/claim_history.csv`:

```csv
claim_id,policy_id,treatment_date,amount,clinic,claim_status,notes
CLM001,POL-1001,2023-01-15,320.50,Pawsome Pet Clinic,APPROVED,Initial X-ray and exam
CLM001,POL-1001,2023-02-01,540.75,Pawsome Pet Clinic,APPROVED,Cast application
CLM001,POL-1001,2023-03-15,180.25,Pawsome Pet Clinic,APPROVED,Cast removal and follow-up
CLM001,POL-1001,2023-04-12,210.00,Pawsome Pet Clinic,APPROVED,Physical therapy
CLM001,POL-1001,2023-05-03,120.50,Pawsome Pet Clinic,APPROVED,Final check-up
CLM002,POL-1002,2023-01-20,450.75,Downtown Animal Hospital,APPROVED,Emergency visit and medication
CLM002,POL-1002,2023-02-10,680.25,Downtown Animal Hospital,APPROVED,Surgery for obstruction
CLM002,POL-1002,2023-03-01,210.50,Downtown Animal Hospital,APPROVED,Post-operative check-up
CLM002,POL-1002,2023-04-15,180.00,Downtown Animal Hospital,APPROVED,Follow-up visit
CLM002,POL-1002,2023-05-10,90.25,Downtown Animal Hospital,APPROVED,Final check-up
CLM003,POL-1003,2023-01-25,280.50,River Valley Veterinary Clinic,APPROVED,Initial consultation and X-rays
CLM003,POL-1003,2023-02-15,560.25,River Valley Veterinary Clinic,APPROVED,Anti-inflammatory treatment
CLM003,POL-1003,2023-03-10,180.75,River Valley Veterinary Clinic,APPROVED,Follow-up and additional medication
CLM003,POL-1003,2023-04-05,240.00,River Valley Veterinary Clinic,APPROVED,Physical therapy
CLM003,POL-1003,2023-05-20,110.50,River Valley Veterinary Clinic,APPROVED,Final assessment
CLM004,POL-1004,2023-02-01,350.25,Healthy Paws Clinic,APPROVED,Initial dental examination
CLM004,POL-1004,2023-02-20,720.50,Healthy Paws Clinic,APPROVED,Dental surgery
CLM004,POL-1004,2023-03-15,190.75,Healthy Paws Clinic,APPROVED,Post-surgery check-up
CLM004,POL-1004,2023-04-10,160.00,Healthy Paws Clinic,APPROVED,Follow-up examination
CLM004,POL-1004,2023-05-05,100.25,Healthy Paws Clinic,APPROVED,Final dental check-up
CLM005,POL-1005,2023-02-05,420.50,Sunset Veterinary Center,APPROVED,Emergency visit after accident
CLM005,POL-1005,2023-02-25,860.75,Sunset Veterinary Center,APPROVED,Surgery for broken leg
CLM005,POL-1005,2023-03-20,220.25,Sunset Veterinary Center,APPROVED,Cast check and adjustment
CLM005,POL-1005,2023-04-15,190.00,Sunset Veterinary Center,APPROVED,Physical therapy initiation
CLM005,POL-1005,2023-05-10,150.50,Sunset Veterinary Center,APPROVED,Cast removal and assessment
``` 