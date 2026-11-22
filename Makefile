.PHONY: setup run serve search evaluate check-data load try-prompts try-xray-words try-crowding smoke clean

# Uniform targets for The Playbooks. The README's commands (pip install -r requirements.txt,
# ./run_pipeline.sh, python test_claims_search.py, python run_tests.py) still work as written.

# Install the locked dependencies into .venv and download the CLIP model (about 600 MB, once).
setup:
	uv sync --frozen
	uv run --frozen python -c "from claims.embeddings import _model; _model()"

# Start Weaviate, embed and load the five sample claims, then run three searches and print
# the results: text to pet photo, pet photo to pet photo, X-ray to X-ray. Exits when done.
run: setup
	docker compose up -d --wait weaviate
	uv run --frozen python store_with_embedding.py
	uv run --frozen python search_similar_claims.py --describe "a photo of a French Bulldog" --limit 2
	uv run --frozen python search_similar_claims.py --photo data/PetPhotos/max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg --limit 2
	uv run --frozen python search_similar_claims.py --xray data/PetXrays/rocky_1.2.410.200028.100.3.20190927.1032580692.2539.1.1.png --limit 2

# Start Weaviate, load the claims, then the embedding API on http://localhost:8001 and the
# Streamlit UI on http://localhost:8501. Ctrl+C stops both; make clean stops Weaviate.
serve: setup
	docker compose up -d --wait weaviate
	uv run --frozen python store_with_embedding.py
	uv run --frozen uvicorn app:app --host 0.0.0.0 --port 8001 & API=$$!; \
	trap "kill $$API" EXIT; \
	uv run --frozen streamlit run streamlit_app.py --server.port 8501 --server.headless true

# With Weaviate running and the claims loaded (make run): search them yourself.
#   make search TEXT="limping after a fall"        claim descriptions
#   make search DESCRIBE="a curly-haired dog"      pet photos, by words
#   make search XRAY=path/to/xray.png              similar X-rays
#   make search PHOTO=path/to/photo.jpg            similar pet photos
search:
	@if [ -n "$(TEXT)" ]; then uv run --frozen python search_similar_claims.py --text "$(TEXT)"; \
	elif [ -n "$(DESCRIBE)" ]; then uv run --frozen python search_similar_claims.py --describe "$(DESCRIBE)"; \
	elif [ -n "$(XRAY)" ]; then uv run --frozen python search_similar_claims.py --xray "$(XRAY)"; \
	elif [ -n "$(PHOTO)" ]; then uv run --frozen python search_similar_claims.py --photo "$(PHOTO)"; \
	else echo 'Usage: make search TEXT="..." | DESCRIBE="..." | XRAY=file | PHOTO=file'; exit 2; fi

# With the claims loaded: how often each kind of search finds the right claim (python run_tests.py).
evaluate:
	uv run --frozen python -m claims.evaluate

# Your own claims, in a folder laid out like data/ (see README, "Use your own data").
# check-data checks every file and sends nothing; load replaces the claims in Weaviate with yours.
#   make check-data DIR=my_claims
#   make load DIR=my_claims
check-data:
	@test -n "$(DIR)" || { echo "Usage: make check-data DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/own_claims.py --dir "$(DIR)" --check-only

load: setup
	@test -n "$(DIR)" || { echo "Usage: make load DIR=path/to/folder"; exit 2; }
	docker compose up -d --wait weaviate
	uv run --frozen python scripts/own_claims.py --dir "$(DIR)"

# Try this: three measured experiments (see README, "Try this"). CLIP only, no Docker.
try-prompts: setup
	uv run --frozen python experiments/prompts.py

try-xray-words: setup
	uv run --frozen python experiments/xray_words.py

try-crowding: setup
	uv run --frozen python experiments/crowding.py

# Tests. The search tests need Weaviate (docker compose up -d --wait weaviate) and skip without it.
smoke:
	uv sync --frozen
	uv run --frozen pytest -q tests/test_claims.py

# Stop Weaviate and delete its volume, and remove generated files. The sample data is kept.
clean:
	-docker compose down -v --remove-orphans
	rm -rf output .pytest_cache
