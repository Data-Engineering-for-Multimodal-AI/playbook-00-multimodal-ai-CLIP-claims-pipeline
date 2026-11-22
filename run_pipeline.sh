#!/bin/bash
# Start Weaviate, set up the schema, start the CLIP embedding API, load the claims, then open the UI.
#   ./run_pipeline.sh            # ... and start Streamlit on http://localhost:8501
#   ./run_pipeline.sh --no-ui    # stop after loading the claims
# PYTHON picks the interpreter (default: python). make run-pipeline uses PYTHON="uv run --frozen python".

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

WEAVIATE_PORT=8080
EMBEDDING_API_PORT=8001
STREAMLIT_PORT=8501
PYTHON="${PYTHON:-python}"

echo -e "=== Starting Multimodal Claims Pipeline ==="

# Check for required Python packages
echo "Checking for required Python packages..."
if ! $PYTHON -c "import fastapi, streamlit, weaviate, transformers, torch" 2>/dev/null; then
    echo -e "${RED}Missing required Python packages. Installing them now...${NC}"
    $PYTHON -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install required packages. Please check your environment.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}Dependencies look good. Continuing setup...${NC}"

# Check if Weaviate is running. The ready endpoint answers HTTP 200 when it is.
weaviate_ready() {
    curl -sf -o /dev/null http://localhost:${WEAVIATE_PORT}/v1/.well-known/ready
}
echo "Checking if Weaviate is running..."
if ! weaviate_ready; then
    echo -e "${YELLOW}Weaviate not running. Starting Weaviate (docker compose)...${NC}"
    if ! docker compose up -d --wait weaviate; then
        echo -e "${RED}Weaviate failed to start. Please check Docker.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}SUCCESS: Weaviate is already running.${NC}"
fi

# Set up Weaviate schema
echo "Setting up Weaviate schema..."
$PYTHON weaviate_schema.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to set up Weaviate schema. Check the error messages above.${NC}"
    exit 1
fi
echo -e "${GREEN}Weaviate schema setup completed.${NC}"

# Check if the embedding API is already running
echo "Starting the CLIP embedding API..."
if ! curl -s http://localhost:${EMBEDDING_API_PORT}/ > /dev/null; then
    # Start the embedding API in the background
    $PYTHON -m uvicorn app:app --host 0.0.0.0 --port ${EMBEDDING_API_PORT} &
    EMBEDDING_API_PID=$!

    # Wait for the embedding API to start (the first run downloads CLIP)
    echo "Waiting for the embedding API to start..."
    COUNTER=0
    while ! curl -s http://localhost:${EMBEDDING_API_PORT}/ > /dev/null; do
        sleep 1
        COUNTER=$((COUNTER+1))
        if [ $COUNTER -gt 600 ]; then
            echo -e "${RED}Embedding API failed to start within 600 seconds. Please check for errors.${NC}"
            kill $EMBEDDING_API_PID 2>/dev/null
            exit 1
        fi
    done
    echo -e "${GREEN}SUCCESS: Embedding API started successfully!${NC}"
else
    echo -e "${GREEN}SUCCESS: Embedding API is already running.${NC}"
fi

trap "echo -e '\nCleaning up...' && kill \$STREAMLIT_PID \$EMBEDDING_API_PID 2>/dev/null; echo -e '${GREEN}SUCCESS: Pipeline shutdown complete!${NC}\nTo run the app again, just run ./run_pipeline.sh'" EXIT

# The other scripts get their embeddings from the API instead of loading CLIP again
export EMBEDDING_API_URL="http://localhost:${EMBEDDING_API_PORT}"

# Ingest claims data
echo "Ingesting claims data into Weaviate..."
$PYTHON store_with_embedding.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Data ingestion failed. Check the logs above.${NC}"
    exit 1
fi
echo -e "${GREEN}SUCCESS: Claims data ingested successfully!${NC}"

# Check that we have data in Weaviate
echo "Checking data in Weaviate..."
CLAIM_COUNT=$($PYTHON -c "
from claims import config, store
client = store.connect()
try:
    print(len(client.collections.get(config.COLLECTION)) if client.collections.exists(config.COLLECTION) else 0)
finally:
    client.close()
")
echo "Found ${CLAIM_COUNT} claims in Weaviate"

if [ "${CLAIM_COUNT:-0}" -eq 0 ]; then
    echo -e "${YELLOW}No claims found in Weaviate. Please run the ingestion script first.${NC}"
fi

if [ "${1:-}" = "--no-ui" ]; then
    exit 0
fi

# Start the Streamlit demo app
echo "Starting the Streamlit demo app..."
echo "Access the app at http://localhost:${STREAMLIT_PORT}"
$PYTHON -m streamlit run streamlit_app.py --server.port ${STREAMLIT_PORT} &
STREAMLIT_PID=$!

# Give control back to the user
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the pipeline${NC}"

wait $STREAMLIT_PID
