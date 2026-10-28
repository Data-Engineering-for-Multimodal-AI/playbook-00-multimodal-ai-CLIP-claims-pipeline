#!/usr/bin/env bash
# Start Weaviate, load the claims, measure the search, then open the search UI.
#   ./run_pipeline.sh            # ... and start Streamlit on http://localhost:8501
#   ./run_pipeline.sh --no-ui    # stop after the evaluation
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

echo "=== Starting Weaviate (docker compose) ==="
docker compose up -d --wait weaviate

echo "=== Loading the claims ==="
"$PYTHON" store_with_embedding.py

echo "=== How well does the search work? ==="
"$PYTHON" -m claims.evaluate

if [[ "${1:-}" != "--no-ui" ]]; then
  echo "=== Search UI: http://localhost:8501 (Ctrl+C to stop; docker compose down stops Weaviate) ==="
  "$PYTHON" -m streamlit run streamlit_app.py --server.address 127.0.0.1
fi
