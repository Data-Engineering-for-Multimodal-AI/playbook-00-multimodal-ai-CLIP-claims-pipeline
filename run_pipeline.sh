#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

WEAVIATE_PORT=8080
EMBEDDING_API_PORT=8001
STREAMLIT_PORT=8501

echo -e "=== Starting Multimodal Claims Pipeline ==="

# Check for required Python packages
echo "Checking for required Python packages..."
if ! pip list | grep -q "fastapi\|streamlit\|weaviate-client\|transformers\|torch"; then
    echo -e "${RED}Missing required Python packages. Installing them now...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install required packages. Please check your environment.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}Dependencies look good. Continuing setup...${NC}"

# Check if Weaviate is running
echo "Checking if Weaviate is running..."
if ! curl -s http://localhost:${WEAVIATE_PORT}/v1/.well-known/ready | grep -q "true"; then
    echo -e "${YELLOW}Weaviate not running. Starting Weaviate container...${NC}"
    docker run -d --name weaviate -p ${WEAVIATE_PORT}:8080 \
        --env QUERY_DEFAULTS_LIMIT=25 \
        --env PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
        --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        --env DEFAULT_VECTORIZER_MODULE=none \
        --restart unless-stopped \
        semitechnologies/weaviate:1.23.9
    
    # Wait for Weaviate to start
    echo "Waiting for Weaviate to start..."
    COUNTER=0
    while ! curl -s http://localhost:${WEAVIATE_PORT}/v1/.well-known/ready | grep -q "true"; do
        sleep 1
        COUNTER=$((COUNTER+1))
        if [ $COUNTER -gt 60 ]; then
            echo -e "${RED}Weaviate failed to start within 60 seconds. Please check Docker.${NC}"
            exit 1
        fi
    done
else
    echo -e "${GREEN}SUCCESS: Weaviate is already running.${NC}"
fi

# Set up Weaviate schema
echo "Setting up Weaviate schema..."
python weaviate_schema.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to set up Weaviate schema. Check the error messages above.${NC}"
    exit 1
fi
echo -e "${GREEN}Weaviate schema setup completed.${NC}"

# Check if the embedding API is already running
echo "Starting the CLIP embedding API..."
if ! curl -s http://localhost:${EMBEDDING_API_PORT}/ > /dev/null; then
    # Start the embedding API in the background
    uvicorn app:app --host 0.0.0.0 --port ${EMBEDDING_API_PORT} &
    EMBEDDING_API_PID=$!
    
    # Wait for the embedding API to start
    echo "Waiting for the embedding API to start..."
    COUNTER=0
    while ! curl -s http://localhost:${EMBEDDING_API_PORT}/ > /dev/null; do
        sleep 1
        COUNTER=$((COUNTER+1))
        if [ $COUNTER -gt 60 ]; then
            echo -e "${RED}Embedding API failed to start within 60 seconds. Please check for errors.${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}SUCCESS: Embedding API started successfully!${NC}"
else
    echo -e "${GREEN}SUCCESS: Embedding API is already running.${NC}"
fi

# Ingest claims data
echo "Ingesting claims data into Weaviate..."
python store_with_embedding.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Data ingestion may have encountered issues. Check the logs above.${NC}"
else
    echo -e "${GREEN}SUCCESS: Claims data ingested successfully!${NC}"
fi

# Check that we have data in Weaviate
echo "Checking data in Weaviate..."
CLAIM_COUNT=$(python -c "import weaviate; client = weaviate.Client('http://localhost:${WEAVIATE_PORT}'); print(client.query.aggregate('ClaimDocument').with_fields('meta { count }').do()['data']['Aggregate']['ClaimDocument'][0]['meta']['count'] if 'data' in client.query.aggregate('ClaimDocument').with_fields('meta { count }').do() and 'Aggregate' in client.query.aggregate('ClaimDocument').with_fields('meta { count }').do()['data'] and 'ClaimDocument' in client.query.aggregate('ClaimDocument').with_fields('meta { count }').do()['data']['Aggregate'] and len(client.query.aggregate('ClaimDocument').with_fields('meta { count }').do()['data']['Aggregate']['ClaimDocument']) > 0 else 0)")
echo "Found ${CLAIM_COUNT} claims in Weaviate"

if [ "$CLAIM_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No claims found in Weaviate. Please run the ingestion script first.${NC}"
fi

# Start the Streamlit demo app
echo "Starting the Streamlit demo app..."
echo "Access the app at http://localhost:${STREAMLIT_PORT}"
streamlit run streamlit_app.py &
STREAMLIT_PID=$!

# Give control back to the user
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the pipeline${NC}"

# Wait for user to press Ctrl+C
trap "echo -e '\nCleaning up...' && kill $STREAMLIT_PID $EMBEDDING_API_PID 2>/dev/null; echo -e '${GREEN}SUCCESS: Pipeline shutdown complete!${NC}\nTo run the app again, just run ./run_pipeline.sh'" EXIT

wait $STREAMLIT_PID 