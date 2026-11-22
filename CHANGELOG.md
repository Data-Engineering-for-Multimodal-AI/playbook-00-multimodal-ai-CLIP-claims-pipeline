# Changelog

## Unreleased (v1.1.0)

Not printed in the book.

### Added
- `make search TEXT=... | DESCRIBE=... | XRAY=... | PHOTO=...` and `make evaluate`.
- `make check-data DIR=...` and `make load DIR=...` (`scripts/own_claims.py`): check a folder of
  your own claims in the layout of `data/`, and load it into Weaviate. Tests in
  `tests/test_own_data.py`; `make smoke` runs `tests/test_claims.py` as before.
- Three measured experiments, CLIP only, no Docker: `make try-prompts`, `make try-xray-words`,
  `make try-crowding` (`experiments/`).
- README: Run it, time tiers, Try this, Use your own data, What you end up with, Before production,
  What it costs.
- CI: the Run it steps with checks on their output, the own-data checks, the three experiments, and
  the time to first output in the devcontainer job.
- `make setup`, `make run`, `make serve`, `make smoke` and `make clean`. `make run` starts Weaviate,
  loads the five sample claims and prints three searches (text to pet photo, pet photo, X-ray);
  `make serve` starts the embedding API and the Streamlit UI.
- `pyproject.toml` and `uv.lock` (Python 3.12 or newer). `requirements.txt` is now generated from the
  lock, with pinned versions.
- `docker-compose.yml` for Weaviate 1.40.2.
- Tests in `tests/`.
- `.devcontainer/` so the repo opens in GitHub Codespaces, with Docker-in-Docker for Weaviate.
- CI: Python 3.12 and 3.13 with Weaviate in Docker, the `pip install -r requirements.txt` path,
  a weekly scheduled run, a `make run` step, and a job that builds the devcontainer and runs
  `make run` inside it.

### Fixed
These fixes are copied from the author's `update-2026` branch (PR #1).
- `run_pipeline.sh` checked Weaviate's ready endpoint for the text `true`, which it never returns.
  It now checks the HTTP status.
- Ingestion stored `claim_note` and `xray_image`, but search asked for `description` and
  `image_path`, and each claim had only its text vector. Image searches failed and fell back to
  keyword matching without saying so. The scripts now share one schema with three named vectors
  per claim (claim text, X-ray, pet photo), using the Weaviate v4 client.
- The embedding API crashed on transformers 5 and averaged unnormalized text and image vectors.
- Three pet photos were filed under the wrong pets. They are renamed to match the breeds in the
  policies.
- `test_streamlit_app.py` needs selenium, which is not in the requirements. It now says so and exits
  instead of failing with an import error.

### Changed
- `claims/store.py` stores image paths outside the repository as absolute paths (needed for
  `make load`); the sample's paths are unchanged.
- README: "Overview" is now "How it works"; "Setup Instructions" is now "Run it the way the book
  does", with the same content.
- README opening: a title that says what you'll do, three sentences on why, the example and what you'll do, then the data engineering problem and what you'll work with.
- `run_pipeline.sh` starts Weaviate with `docker compose` instead of `docker run`, and takes
  `--no-ui` to stop after loading the claims.
- `run_tests.py` runs the retrieval evaluation (`python -m claims.evaluate`).
- `search_similar_claims.py` and `test_claims_search.py` take `--text`, `--describe`, `--xray`,
  `--photo` and `--keywords`.
- The Streamlit app is the version from `update-2026` (four search modes).
