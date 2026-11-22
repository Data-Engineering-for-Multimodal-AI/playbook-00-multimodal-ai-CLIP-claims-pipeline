"""
The claims collection in Weaviate, and loading the sample claims into it.

Each claim is one object with its policy, history and notes as properties, and three named vectors:
  text   the claim description (CLIP text embedding)
  xray   the claim's X-ray (CLIP image embedding)
  photo  the pet's photo (CLIP image embedding)
Searching by X-ray compares X-rays with X-rays; searching by photo, photos with photos; a text query
can search the claim descriptions, or the photos (CLIP puts text and images in one space).
"""
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from claims import config
from claims.embeddings import embed_image, embed_text

VECTORS = ("text", "xray", "photo")


def connect() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host=config.WEAVIATE_HOST, port=config.WEAVIATE_PORT,
                                     grpc_port=config.WEAVIATE_GRPC_PORT)


def create_collection(client: weaviate.WeaviateClient, replace: bool = True):
    if client.collections.exists(config.COLLECTION):
        if not replace:
            return client.collections.get(config.COLLECTION)
        client.collections.delete(config.COLLECTION)
    text = DataType.TEXT
    return client.collections.create(
        config.COLLECTION,
        description="Pet-insurance claim: notes, X-ray, pet photo, policy and history",
        properties=[
            Property(name="claim_id", data_type=text),
            Property(name="pet_name", data_type=text),
            Property(name="breed", data_type=text),
            Property(name="clinic", data_type=text),
            Property(name="policy_id", data_type=text),
            Property(name="claim_note", data_type=text),
            Property(name="vet_note", data_type=text),
            Property(name="xray_path", data_type=text, skip_vectorization=True),
            Property(name="photo_path", data_type=text, skip_vectorization=True),
            Property(name="deductible", data_type=DataType.NUMBER),
            Property(name="co_pay_percent", data_type=DataType.NUMBER),
            Property(name="xray_covered", data_type=DataType.BOOL),
            Property(name="surgery_covered", data_type=DataType.BOOL),
            Property(name="dental_covered", data_type=DataType.BOOL),
            Property(name="claim_history", data_type=text),
            Property(name="total_claimed", data_type=DataType.NUMBER),
        ],
        vector_config=[Configure.Vectors.self_provided(
            name=name, vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE))
            for name in VECTORS],
    )


def _file_for(folder: Path, pet: str) -> Optional[Path]:
    matches = sorted(p for p in folder.iterdir() if p.name.lower().startswith(pet.lower() + "_"))
    return matches[0] if matches else None


def load_claims() -> List[Dict]:
    """The sample claims, one record per claim, with the file each vector comes from."""
    policies = pd.read_csv(config.POLICIES)
    history = pd.read_csv(config.HISTORY)
    claims = []
    for row in policies.to_dict("records"):
        claim_id, pet = row["claim_id"], row["pet_name"]
        events = history[history["claim_id"] == claim_id]
        claims.append({
            "claim_id": claim_id, "pet_name": pet, "breed": row["breed"], "clinic": row["clinic"],
            "policy_id": row["policy_id"],
            "claim_note": (config.CLAIM_NOTES / f"{claim_id}_description.txt").read_text().strip(),
            "vet_note": (config.VET_NOTES / f"{claim_id}_vet_notes.txt").read_text().strip(),
            "xray_path": str(_file_for(config.XRAYS, pet).relative_to(config.ROOT)),
            "photo_path": str(_file_for(config.PHOTOS, pet).relative_to(config.ROOT)),
            "deductible": float(row["deductible"]), "co_pay_percent": float(row["co_pay_percent"]),
            "xray_covered": bool(row["xray_covered"]), "surgery_covered": bool(row["surgery_covered"]),
            "dental_covered": bool(row["dental_covered"]),
            "claim_history": events[["date", "type", "amount"]].to_json(orient="records"),
            "total_claimed": float(events["amount"].sum()),
        })
    return claims


def ingest(client: weaviate.WeaviateClient, verbose: bool = True) -> int:
    collection = create_collection(client)
    claims = load_claims()
    with collection.batch.fixed_size(batch_size=50) as batch:
        for claim in claims:
            vectors = {"text": embed_text(claim["claim_note"]),
                       "xray": embed_image(config.ROOT / claim["xray_path"]),
                       "photo": embed_image(config.ROOT / claim["photo_path"])}
            batch.add_object(properties=claim, vector=vectors,
                             uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"claims/{claim['claim_id']}"))
            if verbose:
                print(f"  {claim['claim_id']} {claim['pet_name']:<6} ({claim['breed']}): note, X-ray and photo embedded")
    failed = collection.batch.failed_objects
    if failed:
        raise RuntimeError(f"{len(failed)} claims failed to load: {failed[0].message}")
    return len(collection)


def history(claim: Dict) -> List[Dict]:
    return json.loads(claim.get("claim_history") or "[]")
