#!/usr/bin/env python3
"""Embed the sample claims with CLIP and load them into Weaviate: python store_with_embedding.py"""
import sys

from claims import config, store

if __name__ == "__main__":
    client = store.connect()
    try:
        if not client.is_ready():
            sys.exit("Weaviate isn't ready: docker compose up -d --wait weaviate")
        print(f"Loading claims into {config.COLLECTION} (CLIP model {config.CLIP_MODEL})...")
        count = store.ingest(client)
        print(f"{count} claims in Weaviate")
    finally:
        client.close()
