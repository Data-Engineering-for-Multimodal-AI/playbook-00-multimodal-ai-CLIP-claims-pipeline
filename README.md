# Playbook 00: Search pet insurance claims by X-ray, text or photo
**How to build multimodal search with CLIP.** A companion playbook for [*Data Engineering for Multimodal AI*](https://oreil.ly/data-engineering-multimodal) (O'Reilly, 2026). Not printed in the book.

A pet insurance claim comes with X-rays, a written description and a policy record, usually kept in different places. In this repo, we explore this with a sample set of pet claims. Follow along to put images and text into one vector space with CLIP, and search claims by X-ray, by symptom description or by a photo of the pet.

**The problem.** Images and text usually live in separate indexes, so you can't search one with the other.

**What you'll work with.** You'll put images and text into one vector space with CLIP, search it, and join the results with structured policy data.

> The original steps are under [Run it the way the book does](#run-it-the-way-the-book-does) and still work as written. Want the repository as it was first published? Use the `first-edition` tag.

## Run it

You need [uv](https://docs.astral.sh/uv/), `make` and Docker (for Weaviate, the vector database). Or open the repo in a GitHub Codespace, where all of them are ready. On Windows without `make`, use a Codespace or the original steps below.

**1. Get the code.** The clone is about 72 MB, most of it the sample X-rays and photos.

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-00-multimodal-ai-CLIP-claims-pipeline.git
cd playbook-00-multimodal-ai-CLIP-claims-pipeline
```

**2. Load the claims and search them.** Installs the locked dependencies (`uv sync --frozen`), downloads the CLIP model once (about 600 MB), starts Weaviate (`docker compose up -d --wait weaviate`), embeds and loads the five sample claims (`python store_with_embedding.py`), then runs three searches (`python search_similar_claims.py`): words to pet photo, pet photo to pet photo, and X-ray to X-ray. On a busy Mac, with the CLIP model and the Weaviate image already downloaded, this took 3 min 23 s.

```bash
make run
```

You should see the five claims load, then each search's two best matches. The first:

```
  CLM005 Rocky  (Jack Russell Terrier): note, X-ray and photo embedded
5 claims in Weaviate
...
Searching with description "a photo of a French Bulldog"

1. CLM002 Jack (French Bulldog, Willow Veterinary Center): similarity 0.305
    policy POL-1002: covers xray; deductible $110, co-pay 20%; claimed so far $725
```

The sentence found Jack's photo: text and images are in one vector space. Each result also carries the claim's policy and history, joined from `data/StructuredData/`. The two image searches find the query image itself first, at similarity 1.000; the runner-up X-ray scores 0.971, the runner-up photo 0.860.

If `docker compose up` fails with `The container name "/claims-weaviate" is already in use`, another copy of this repo has a Weaviate container. Stop and remove that one first (`docker compose down` in its folder).

**3. Search yourself.** With Weaviate still running. By the claim text, or by describing the pet in words:

```bash
make search TEXT="limping after a fall"
make search DESCRIBE="a curly-haired dog"
```

```
Searching with description "a curly-haired dog"

1. CLM001 Emma (Goldendoodle, Good Dog Veterinary Clinic): similarity 0.285
2. CLM004 Otis (Border Collie, Pawsitive Pet Care): similarity 0.268
3. CLM005 Rocky (Jack Russell Terrier, Pinewood Animal Hospital): similarity 0.231
```

(Policy lines left out.) Emma is the Goldendoodle. `XRAY=` and `PHOTO=` take any image file on your machine.

**4. Measure the search.** Runs the retrieval evaluation (`python -m claims.evaluate`, the same as `python run_tests.py`): vet notes against claim descriptions, breeds against photos, and edited images against the originals. This took 25 s on the same Mac.

```bash
make evaluate
```

```
Test                                 Queries   Top-1    MRR
vet note -> claim (CLIP text)              5    0.20   0.52
vet note -> claim (BM25 keywords)          5    0.60   0.60
breed -> photo (text to image)             5    1.00   1.00
edited photo -> photo                     25    1.00   1.00
edited X-ray -> X-ray                     25    0.92   0.95
```

CLIP finds pets by photo and by breed every time, but matches a vet's notes to the right claim only once in five; keyword search does better on those. The misses are listed below the table.

**5. The search UI (optional).** Starts the embedding API on port 8001 and the Streamlit UI on http://localhost:8501 (`uvicorn app:app`, `streamlit run streamlit_app.py`). Ctrl+C stops both. We didn't time this step.

```bash
make serve
```

**6. Clean up.**

```bash
make clean
```

This stops Weaviate and deletes its volume (the loaded claims), and removes `output/` and the test cache. The CLIP model in `~/.cache/huggingface` (1.1 GB on our Mac) and the sample data stay.

## How much time do you have?

| Time | What to do |
|---|---|
| 15 minutes | [Run it](#run-it), steps 1 to 3. On a busy Mac, step 2 took 3 min 23 s with the model already downloaded. |
| 1 hour | Step 4, then the three [Try this](#try-this) experiments (20 to 50 s each on the same Mac). |
| 1 day | [Use your own data](#use-your-own-data) with your own claims, and work through [Before production](#before-production). |

## Try this

Each experiment changes one thing and measures the effect. They embed the five sample claims with CLIP in the script and rank them by cosine similarity, the metric the Weaviate collection uses, so they need no Docker. The numbers below are from a Mac on 2026-11-21.

**1. Which words find a pet's photo?** Builds one query per claim from a template (the breed, the pet's name, the claim note) and checks whether it ranks that pet's photo first.

```bash
make try-prompts
```

```
query template              right photo first   margin  example query
{breed}                               5 of 5   +0.060  Labrador Retriever
a photo of a {breed}                  5 of 5   +0.061  a photo of a Labrador Retriever
a photo of a dog                      1 of 5   -0.009  a photo of a dog
{name}                                0 of 5   -0.015  Max
a photo of {name}, a dog              1 of 5   -0.010  a photo of Max, a dog
{note}                                1 of 5   -0.014  Max seemed stiff and hesitant to jump...
```

The breed finds the photo every time; the pet's name and the claim note don't, because neither describes what the photo shows. Text search over images works when the text comes from a field that describes the image, here the breed in `policy_details.csv`. Add your own templates to `TEMPLATES` in `experiments/prompts.py`.

**2. Can words find an X-ray?** Searches the five X-rays with a sentence naming the body part each claim X-rayed (hip, shoulder, hind legs, chest, lower spine), next to breed-to-photo for comparison.

```bash
make try-xray-words
```

```
search                right claim first   margin  spread
body part -> X-ray              2 of 5   -0.006   0.021
breed -> pet photo              5 of 5   +0.061   0.101
```

Two of five, about what guessing gives, and the five X-rays score within 0.021 of each other for any sentence. CLIP learned from web photos and captions, not radiographs. For X-rays, search by image, or use a model trained on medical images. Change the sentences in `BODY_PARTS` and run it again.

**3. How far apart are the claims in each modality?** Measures the average similarity between different claims' X-rays, photos and notes, then edits each image (the edits `make evaluate` uses) and measures the room left between the right claim and the next.

```bash
make try-crowding
```

```
Average similarity between two different claims (1.0 = identical):
  X-ray       0.953
  pet photo   0.805
  claim note  0.824
  vet note    0.827

modality    edit              right first   margin
X-ray       crop 70%               4 of 5   +0.012
X-ray       shrink to 96 px        4 of 5   +0.014
pet photo   crop 70%               5 of 5   +0.115
pet photo   shrink to 96 px        5 of 5   +0.105
```

(Five edits per modality; the table shows two.) Any two X-rays are 0.953 similar, so a crop or a smaller image is enough to swap the ranking; photos keep a margin about ten times larger. With thousands of claims, X-ray search by CLIP would be unreliable. Change `EDITS` in `claims/evaluate.py` to try other edits.

## Use your own data

Put your claims in a folder with the same layout as `data/`:

```
my_claims/
  StructuredData/policy_details.csv   claim_id, policy_id, pet_name, breed, clinic, xray_covered,
                                      surgery_covered, dental_covered (True/False), deductible, co_pay_percent
  StructuredData/claim_history.csv    claim_id, pet_name, date (YYYY-MM-DD), type, amount
  ClaimNotes/<claim_id>_description.txt   UTF-8
  VetNotes/<claim_id>_vet_notes.txt       UTF-8
  PetXrays/<pet name>_<anything>.png      one per pet; .png .jpg .jpeg .webp .bmp
  PetPhotos/<pet name>_<anything>.jpg     one per pet
```

Images are found by pet name (lowercase, then `_`), so each pet name can appear once. Check it first. This reads every file and sends nothing:

```bash
make check-data DIR=my_claims
```

It reports every problem with the file and the reason: a missing file or column, text that isn't UTF-8, an empty note, an image that won't open or is smaller than 32x32, a Git LFS pointer, a coverage flag that isn't True or False, a number or date that doesn't parse, a repeated claim ID or pet name, or a history row for a claim that doesn't exist. Notes longer than CLIP's 77 tokens get a note: CLIP embeds only the start. On a folder with deliberate mistakes:

```
8 problems in my_claims; nothing was loaded:
  VetNotes/CLM003_vet_notes.txt: not UTF-8 text
  StructuredData/policy_details.csv line 4: xray_covered is 'yes', needs True or False
  StructuredData/policy_details.csv line 4: co_pay_percent is 'abc', not a number
  VetNotes/CLM010_vet_notes.txt: missing
  PetXrays/bella_1.png: too small (20x20), needs at least 32x32
  PetPhotos/bella_beach.jpg: a Git LFS pointer, not the image (run git lfs pull)
  StructuredData/claim_history.csv line 12: date is '15/03/2024', needs YYYY-MM-DD
  StructuredData/claim_history.csv line 13: claim_id 'CLM099' isn't in policy_details.csv
```

When everything passes, load it. This starts Weaviate if needed, embeds every claim and replaces the claims in the collection with yours:

```bash
make load DIR=my_claims
make search DESCRIBE="a photo of a Labrador Retriever"
```

```
my_claims: 2 claims OK (note, vet note, X-ray and photo for each)
Loading my_claims into ClaimDocument (replaces the claims there)...
  CLM001 Emma   (Goldendoodle): note, X-ray and photo embedded
  CLM003 Max    (Labrador Retriever): note, X-ray and photo embedded
2 claims in Weaviate. Search them with make search; make run puts the sample back.
```

`make evaluate` and `make serve` then work on your claims too. `make run` (or `python store_with_embedding.py`) loads the sample again.

## What you end up with

A vector search over your claims in which a sentence, an X-ray or a pet photo finds the claim, with its policy and history joined in, and measurements of where that search works and where it doesn't.

Skills you'll practice:
- Embedding text and images into one vector space with CLIP
- Storing several named vectors per record in Weaviate, next to structured fields
- Searching across modalities, and checking which text fields describe the images
- Measuring retrieval with held-out queries (vet notes, edited images)
- Validating input files and joins before they reach the index
- Telling when a general model doesn't fit a modality (X-rays)

## Before production

This playbook runs five claims on one machine. Before you run something like it for real:

1. **Use a model that fits each modality.** CLIP found the right X-ray from words 2 times in 5, and any two X-rays are 0.953 similar ([Try this 2 and 3](#try-this)). Evaluate a model trained on radiographs for the `xray` vector before relying on X-ray search. *Chapter 5, "Extracting Features Across Modalities" (page 139).*
2. **Measure retrieval on a held-out set, at your scale.** `make evaluate` uses five claims, so top-1 is a coarse number. Build a labelled set of queries from real claims and track recall and MRR for each vector as the collection grows. *Chapter 8, "Testing and Validation for Multimodal Models" (page 286).*
3. **Combine keyword and vector search for notes.** BM25 matched vet notes to claims 3 times in 5, CLIP text once ([Run it, step 4](#run-it)). Weaviate supports hybrid queries; measure them against both. *Chapter 6, "Hybrid Storage and Retrieval" (page 189).*
4. **Load incrementally.** `store.ingest` deletes and recreates the collection on every run, and `make load` replaces the sample. Insert and update claims by their UUID (already derived from the claim ID) instead. *Chapter 4, "Operationalizing ELT for Multimodal AI" (page 112).*
5. **Validate at the API too.** `make check-data` checks files on your machine; the embedding API (`app.py`) accepts any image under `data/`. Check type, size and readability where data enters. *Chapter 4, "Validation, Transformation, Observability, and Annotation Across Modalities" (page 119).*
6. **Record which model made each vector.** The collection doesn't store the CLIP model name, and `CLIP_MODEL` can be changed by an environment variable. Store it with every object so a model change means a planned re-embed. *Chapter 8, "Data Versioning and Experiment Tracking for Multimodal AI" (page 276).*
7. **Protect the claims.** Weaviate runs with anonymous access (bound to 127.0.0.1), and the claims hold medical notes and policy data. Add authentication, encryption and access by role. *Chapter 11, "Protecting Data Across the Pipeline" (page 421).*
8. **Name containers per project.** `docker-compose.yml` fixes `container_name: claims-weaviate`, so a second copy of the repo can't start Weaviate (we hit this). Let Compose name it. *Chapter 2, "Containerization and Orchestration for Multimodal AI" (page 43).*
9. **Know where the images came from.** The photos are from Unsplash and the X-rays are described as anonymized samples, without per-file sources or licenses. Record source and license for every file you load. *Chapter 11, "Lineage and Provenance Tracking" (page 416).*

## What it costs

| Where you run it | Cost |
|---|---|
| Your machine | $0. Everything runs locally: no API keys, no paid services. The clone is about 72 MB; the CLIP model cache took 1.1 GB and the Python environment (`.venv`) another 1.1 GB. |
| A GitHub Codespace (2 cores, 8 GB, the default size; this repo doesn't ask for more) | Estimated $0.18 per hour at GitHub's list price (November 2026). A free personal account includes 120 core-hours a month, about 60 hours on this machine. In CI, `make run` in a freshly built devcontainer, where `make setup` has already downloaded the model, took 44 s from start to the last search result. |

## How it works

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

- **Python 3.12+**
- **Docker** - Required for running Weaviate
- **Git** - For cloning the repository

## Run it the way the book does

The book doesn't print this repository; these are the original steps, and they still work as written.

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

Weaviate runs as a Docker container, defined in `docker-compose.yml`. The script will start it, but you can also run it manually:

```bash
docker compose up -d --wait weaviate
```

You can check if Weaviate is running by visiting http://localhost:8080/v1/.well-known/ready in your browser. It answers with HTTP status 200 when it is ready.

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
Collection ClaimDocument created (named vectors: text, xray, photo)
Weaviate schema setup completed.

Starting the CLIP embedding API...
INFO:     Started server process [28842]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
SUCCESS: Embedding API started successfully!

Ingesting claims data into Weaviate...
Loading claims into ClaimDocument (CLIP model openai/clip-vit-base-patch32)...
  CLM001 Emma   (Goldendoodle): note, X-ray and photo embedded
  CLM002 Jack   (French Bulldog): note, X-ray and photo embedded
  CLM003 Max    (Labrador Retriever): note, X-ray and photo embedded
  CLM004 Otis   (Border Collie): note, X-ray and photo embedded
  CLM005 Rocky  (Jack Russell Terrier): note, X-ray and photo embedded
5 claims in Weaviate
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
python test_claims_search.py --photo data/PetPhotos/rocky_victor-g-N04FIfHhv_k-unsplash.jpg

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
