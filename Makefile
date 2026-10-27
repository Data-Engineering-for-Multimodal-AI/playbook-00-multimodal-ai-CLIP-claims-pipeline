.PHONY: smoke weaviate load evaluate ui api run down
smoke:        ## tests (start Weaviate first for the search tests)
	uv run pytest -q tests
weaviate:
	docker compose up -d --wait weaviate
load: weaviate
	uv run python store_with_embedding.py
evaluate: load
	uv run python -m claims.evaluate
ui: load
	uv run streamlit run streamlit_app.py --server.address 127.0.0.1
api:          ## optional embedding API on http://127.0.0.1:8001
	uv run python app.py
run:
	PYTHON="uv run python" ./run_pipeline.sh
down:
	docker compose down
