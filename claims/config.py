"""Paths and connection settings (environment variables override the defaults)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLAIM_NOTES = DATA / "ClaimNotes"
VET_NOTES = DATA / "VetNotes"
XRAYS = DATA / "PetXrays"
PHOTOS = DATA / "PetPhotos"
POLICIES = DATA / "StructuredData" / "policy_details.csv"
HISTORY = DATA / "StructuredData" / "claim_history.csv"

WEAVIATE_HOST = os.environ.get("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.environ.get("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
COLLECTION = "ClaimDocument"
CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
# Set to use the embedding API (python app.py) instead of loading CLIP in this process
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "")
