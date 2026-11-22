.PHONY: setup run serve smoke clean

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

# Tests. The search tests need Weaviate (docker compose up -d --wait weaviate) and skip without it.
smoke:
	uv sync --frozen
	uv run --frozen pytest -q tests

# Stop Weaviate and delete its volume, and remove generated files. The sample data is kept.
clean:
	-docker compose down -v --remove-orphans
	rm -rf output .pytest_cache
