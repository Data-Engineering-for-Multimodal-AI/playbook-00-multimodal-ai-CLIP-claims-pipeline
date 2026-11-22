"""
Search the claims: by text, by X-ray, by pet photo, or by describing the pet in words.

    by_text("limping after a fall")          claim descriptions most like the text
    by_xray("data/PetXrays/...png")         claims whose X-ray looks most like this one
    by_photo("photo.jpg")                    claims whose pet photo looks most like this one
    by_description("a curly-haired dog")     pet photos that match the words (text -> image)
"""
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image
from weaviate.classes.query import MetadataQuery

from claims import config
from claims.embeddings import embed_image, embed_text


def _near(client, vector: List[float], target: str, limit: int) -> List[Dict]:
    collection = client.collections.get(config.COLLECTION)
    response = collection.query.near_vector(near_vector=vector, target_vector=target, limit=limit,
                                            return_metadata=MetadataQuery(distance=True))
    return [{**o.properties, "similarity": round(1 - o.metadata.distance, 4), "matched_on": target}
            for o in response.objects]


def by_text(client, text: str, limit: int = 5) -> List[Dict]:
    return _near(client, embed_text(text), "text", limit)


def by_description(client, text: str, limit: int = 5) -> List[Dict]:
    return _near(client, embed_text(text), "photo", limit)


def by_xray(client, image: Union[str, Path, Image.Image], limit: int = 5) -> List[Dict]:
    return _near(client, embed_image(image), "xray", limit)


def by_photo(client, image: Union[str, Path, Image.Image], limit: int = 5) -> List[Dict]:
    return _near(client, embed_image(image), "photo", limit)


def keyword(client, text: str, limit: int = 5, fields=("claim_note", "vet_note")) -> List[Dict]:
    """BM25 keyword search over the notes, for comparison with the embedding search."""
    collection = client.collections.get(config.COLLECTION)
    response = collection.query.bm25(query=text, query_properties=list(fields), limit=limit,
                                     return_metadata=MetadataQuery(score=True))
    return [{**o.properties, "score": round(o.metadata.score, 4), "matched_on": "keywords"} for o in response.objects]


def describe(result: Dict) -> str:
    covered = [k.replace("_covered", "") for k in ("xray_covered", "surgery_covered", "dental_covered") if result.get(k)]
    score = f"similarity {result['similarity']:.3f}" if "similarity" in result else f"BM25 {result['score']:.2f}"
    return (f"{result['claim_id']} {result['pet_name']} ({result['breed']}, {result['clinic']}): {score}\n"
            f"    policy {result['policy_id']}: covers {', '.join(covered) or 'none of x-ray/surgery/dental'}; "
            f"deductible ${result['deductible']:.0f}, co-pay {result['co_pay_percent']:.0f}%; "
            f"claimed so far ${result['total_claimed']:.0f}\n"
            f"    \"{result['claim_note']}\"")
