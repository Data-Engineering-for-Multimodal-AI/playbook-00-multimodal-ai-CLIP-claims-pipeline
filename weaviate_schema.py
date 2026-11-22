#!/usr/bin/env python3
"""Create (or re-create) the ClaimDocument collection in Weaviate: python weaviate_schema.py"""
from claims import config, store

if __name__ == "__main__":
    client = store.connect()
    try:
        store.create_collection(client)
        print(f"Collection {config.COLLECTION} created (named vectors: {', '.join(store.VECTORS)})")
    finally:
        client.close()
