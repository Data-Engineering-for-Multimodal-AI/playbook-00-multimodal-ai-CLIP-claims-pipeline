# Multimodal Claims Search with CLIP and Weaviate

A companion project for *Data Engineering for Multimodal AI* (O'Reilly, 2026). It isn't printed in the book; it puts Chapter 1's idea (several modalities of one case linked into one searchable record) into a small claims-search system.
Book: https://oreil.ly/data-engineering-multimodal

> The version before the 2026 update is tagged `first-edition`; see [Differences from the first edition](#differences-from-the-first-edition).

## What you'll build

Search over five fictional pet-insurance claims, each with a claim note, a vet note, a chest X-ray, a photo of the pet, a policy and a payment history.

- **One embedding space for text and images.** [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (ViT-B/32) turns text and images into 512-number vectors, normalized, so that a sentence and a photo it describes are close together.
- **One record per claim, three vectors.** Weaviate stores each claim with its policy and history as properties and three named vectors: the claim note, the X-ray and the pet photo.
- **Four ways to search**: by text (claim notes), by describing the pet (text against photos), by X-ray, by pet photo. BM25 keyword search is there for comparison.
- **A measured evaluation** (`python -m claims.evaluate`) of what works and what doesn't.
- **A Streamlit app** showing each result's photo, X-ray, notes, coverage and history, and an optional embedding API (FastAPI).

What it shows, measured below:

- **Text-to-image works where CLIP is strong.** "A photo of a French Bulldog" finds Jack's photo; all five breeds are found first.
- **CLIP is a weak text-to-text matcher.** Matching each vet's note to its claim note, CLIP ranks the right claim first in 1 case of 5, and pulls most queries towards one claim. Plain keyword search does better (3 of 5). A sentence-embedding model would be the better choice for the text vector.
- **CLIP isn't a medical model.** All five X-rays are chest radiographs, and to CLIP they look alike: a cropped or shrunken X-ray is still found 23 times out of 25, but by appearance, not by diagnosis.

## Quick start

You need Python 3.12 or newer and Docker.

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-00-multimodal-ai-CLIP-claims-pipeline.git
cd playbook-00-multimodal-ai-CLIP-claims-pipeline

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

./run_pipeline.sh          # starts Weaviate, loads the claims, runs the evaluation, opens the app
```

`./run_pipeline.sh --no-ui` stops after the evaluation. The first run downloads CLIP (600 MB). With [uv](https://docs.astral.sh/uv/): `uv sync`, then `make run`. `docker compose down` stops Weaviate.

Step by step:

```bash
docker compose up -d --wait weaviate           # Weaviate 1.40.2 on 127.0.0.1:8080 (gRPC 50051)
python store_with_embedding.py                 # embed and load the claims
python -m claims.evaluate                      # or: python run_tests.py
streamlit run streamlit_app.py                 # http://localhost:8501
```

Searching from the command line (`test_claims_search.py` takes the same options):

```bash
python search_similar_claims.py --text "persistent cough"
python search_similar_claims.py --describe "a curly-haired dog on a bed"
python search_similar_claims.py --xray data/PetXrays/<file>.png
python search_similar_claims.py --photo data/PetPhotos/<file>.jpg
python search_similar_claims.py --keywords "radiograph shoulder"
```

The embedding API is optional: `python app.py` serves `POST /embed` (form field `text`, file `image`) on http://127.0.0.1:8001; set `EMBEDDING_API_URL=http://127.0.0.1:8001` and the other scripts use it instead of loading CLIP themselves. Settings are in `.env.example`.

## Tested on

| Date | OS | Python | What |
|---|---|---|---|
| 2026-10-27 | macOS 26.5.1 (Apple M3) | 3.13.16 | `make smoke` (4 tests, Weaviate in Docker); the Streamlit app, all four modes, with Streamlit's test runner |
| 2026-10-27 | macOS 26.5.1 (Apple M3) | 3.12.15 | `pip install -r requirements.txt` (63 s), `./run_pipeline.sh --no-ui` (62 s, CLIP already downloaded), the searches below |
| 2026-10-27 | Ubuntu (GitHub Actions) | 3.12, 3.13 | `make smoke`; `pip install -r requirements.txt`, then `./run_pipeline.sh --no-ui` and the scripts |

Versions are pinned in `uv.lock` and `requirements.txt` (weaviate-client 4.24, torch 2.14 (CPU build on Linux), transformers 5.21, Streamlit 1.66).

## Differences from the first edition

| First edition | Now | Why |
|---|---|---|
| Weaviate client 3 (`weaviate.Client`), Weaviate 1.23.9 | Client 4 (`connect_to_local`, collections), Weaviate 1.40.2 | The v3 API is removed in client 4 |
| Ingestion stored `claim_note` and `xray_image`; search asked for `description` and `image_path` | One schema, used by both | The searches failed; the errors were caught and the app fell back to keyword matching |
| Every claim stored with its text vector only, even when an image was sent | Three named vectors per claim: note, X-ray, photo | X-ray and photo searches compared images with text |
| `/embed` averaged unnormalized text and image vectors; crashed on transformers 5 | Normalized vectors; text and image returned separately | Fair cosine similarity; current transformers returns an output object |
| Three pet photos filed under the wrong pets (Jack's file showed a Jack Russell, Otis's a French Bulldog, Rocky's a Border Collie type) | Renamed to match the breeds in the policies | Photo search returned the wrong pet |
| `data/README.md`: Emma dental, Jack digestive, Max fracture, Otis eye infection | What the notes say (hip, shoulder, hind limbs, cough, lumbar spine) | Accuracy |
| `run_pipeline.sh` started Weaviate with `docker run` | `docker compose` (Weaviate on 127.0.0.1) | Repeatable, local only |
| Embedding API on 0.0.0.0; `/embed_file` read any path | 127.0.0.1 by default; files under `data/` only | Not reachable or readable from outside by default |
| Clone URL `pb1-mmclaimspipeline`, Python 3.8+, unpinned requirements | This repository's URL, Python 3.12+, pinned | |
| `demo.py`, `simple_test_app.py` and four manual test scripts for the v3 client | Removed; `pytest` and `python -m claims.evaluate` | `test_claims_search.py` and `run_tests.py` still work |

Full list: [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

| Data | Source | License |
|---|---|---|
| Claim notes, vet notes, policies, claim history, invoices | Made up for this project (fictional pets and clinics) | MIT, like the code |
| `data/PetPhotos/` (5) | [Unsplash](https://unsplash.com), by Brooke Cagle, Karsten Winegeart, Taylor Kopel, Pauline Loroy and Victor G (IDs in the file names and in [data/README.md](data/README.md)) | Unsplash License |
| `data/PetXrays/` (5): lateral chest radiographs of dogs | A veterinary X-ray dataset on Kaggle; **which one is still to be confirmed** | To be confirmed with the source |
| CLIP ViT-B/32 (downloaded on first use) | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | MIT |

## Expected output

From `./run_pipeline.sh --no-ui` and the searches on 2026-10-27, macOS 26.5.1 (Apple M3), Python 3.12.15, after `pip install -r requirements.txt`. This is the output of this version.

Loading:

```
Loading claims into ClaimDocument (CLIP model openai/clip-vit-base-patch32)...
  CLM001 Emma   (Goldendoodle): note, X-ray and photo embedded
  CLM002 Jack   (French Bulldog): note, X-ray and photo embedded
  CLM003 Max    (Labrador Retriever): note, X-ray and photo embedded
  CLM004 Otis   (Border Collie): note, X-ray and photo embedded
  CLM005 Rocky  (Jack Russell Terrier): note, X-ray and photo embedded
5 claims in Weaviate
```

The evaluation (`python -m claims.evaluate`). Top-1: the right claim came first; MRR: mean reciprocal rank.

```
Test                                 Queries   Top-1    MRR
vet note -> claim (CLIP text)              5    0.20   0.52
vet note -> claim (BM25 keywords)          5    0.60   0.60
breed -> photo (text to image)             5    1.00   1.00
edited photo -> photo                     25    1.00   1.00
edited X-ray -> X-ray                     25    0.92   0.95

Misses (query: expected -> found first):
  vet note CLM001: CLM001 -> CLM003 (rank 2)
  vet note CLM002: CLM002 -> CLM003 (rank 4)
  vet note CLM004: CLM004 -> CLM003 (rank 2)
  vet note CLM005: CLM005 -> CLM003 (rank 3)
  vet note CLM001: CLM001 -> CLM004 (rank not found)
  vet note CLM005: CLM005 -> CLM004 (rank not found)
  CLM002 X-ray, shrink to 96 px: CLM002 -> CLM001 (rank 4)
  CLM004 X-ray, crop 70%: CLM004 -> CLM001 (rank 2)
```

Searches:

```
Searching with text "persistent cough"

1. CLM004 Otis (Border Collie, Pawsitive Pet Care): similarity 0.783
    policy POL-1004: covers xray, dental; deductible $130, co-pay 30%; claimed so far $775
    "Otis developed a persistent cough. Clinic advised a chest X-ray to rule out fluid build-up or lung issues."

2. CLM003 Max (Labrador Retriever, River Valley Veterinary Clinic): similarity 0.711
    policy POL-1003: covers xray, surgery; deductible $120, co-pay 25%; claimed so far $750
    "Max seemed stiff and hesitant to jump. Vet visit suggested arthritis or muscle strain. Imaging requested for confirmation."
```

```
Searching with description "a curly-haired dog on a bed"

1. CLM001 Emma (Goldendoodle, Good Dog Veterinary Clinic): similarity 0.319
    policy POL-1001: covers xray, surgery, dental; deductible $100, co-pay 15%; claimed so far $700
    "Emma was limping after playing at the park. We brought her in to rule out a sprain or fracture. The vet recommended an X-ray to check her hip."

2. CLM004 Otis (Border Collie, Pawsitive Pet Care): similarity 0.234
    policy POL-1004: covers xray, dental; deductible $130, co-pay 30%; claimed so far $775
    "Otis developed a persistent cough. Clinic advised a chest X-ray to rule out fluid build-up or lung issues."
```

```
Searching with X-ray data/PetXrays/otis_1.2.410.200028.100.3.20190926.1008510164.10580.1.2.png

1. CLM004 Otis (Border Collie, Pawsitive Pet Care): similarity 1.000
    policy POL-1004: covers xray, dental; deductible $130, co-pay 30%; claimed so far $775
    "Otis developed a persistent cough. Clinic advised a chest X-ray to rule out fluid build-up or lung issues."

2. CLM001 Emma (Goldendoodle, Good Dog Veterinary Clinic): similarity 0.963
    policy POL-1001: covers xray, surgery, dental; deductible $100, co-pay 15%; claimed so far $700
    "Emma was limping after playing at the park. We brought her in to rule out a sprain or fracture. The vet recommended an X-ray to check her hip."
```

The X-ray search finds Otis's own X-ray (similarity 1.000), but Emma's chest X-ray is a close second (0.963): to CLIP all five chest radiographs look much alike.

## License

MIT; see [LICENSE](LICENSE). The photos are under the Unsplash License; see [Data and licenses](#data-and-licenses).
