# Changelog

## 2026.10 (2026-10-27)

The version before this update is tagged `first-edition`.

### Fixed
- Search didn't work: ingestion stored `claim_note`/`xray_image` while search asked for
  `description`/`image_path`, and every claim was stored with its text vector only, so X-ray and photo
  searches compared images with text. Errors were swallowed.
- The embedding API crashed on current transformers (feature methods now return an output object), and
  averaged unnormalized text and image vectors.
- Three pet photos were filed under the wrong pets (their breeds didn't match); renamed.
- The data README described dental, digestive and eye problems that the notes don't contain.

### Changed
- Weaviate client 4 (the v3 `weaviate.Client` API is gone) and Weaviate 1.40.2 (was 1.23.9). One
  collection with three named vectors: claim text, X-ray, pet photo.
- New search by describing the pet (text to photo), and BM25 keyword search for comparison.
- `python -m claims.evaluate` measures retrieval; `run_pipeline.sh` runs it.
- Streamlit app rewritten (four search modes; photo, X-ray, policy and history per result).
- The embedding API listens on 127.0.0.1 and only reads files under `data/`.
- Pinned dependencies (`uv.lock`, `requirements.txt`), Python 3.12+; `docker-compose.yml` for Weaviate.
- Tests and CI (Python 3.12 and 3.13, Weaviate in Docker).

### Removed
- `demo.py`, `simple_test_app.py` and four manual test scripts written for the v3 client;
  `pytest` and `python -m claims.evaluate` replace them. `test_claims_search.py` and `run_tests.py` remain.
