"""
CLIP embeddings for text and images, L2-normalized so that cosine distance compares them fairly.

CLIP maps text and images into one 512-dimensional space: a sentence and a photo it describes land
close together. That is what lets a text query find a pet photo. CLIP was trained on web images, not
radiographs; on X-rays it measures visual similarity, not medical findings.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Union

import numpy as np
import requests
from PIL import Image

from claims import config


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    torch.set_grad_enabled(False)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL).eval()
    return model, CLIPProcessor.from_pretrained(config.CLIP_MODEL)


def _vector(features) -> np.ndarray:
    # transformers 5 returns an output object; the projected embedding is its pooler_output
    tensor = getattr(features, "pooler_output", features)
    v = tensor.detach().cpu().numpy()[0].astype(np.float32)
    return v / np.linalg.norm(v)


def embed_text(text: str) -> List[float]:
    if config.EMBEDDING_API_URL:
        return _remote({"text": text})
    model, processor = _model()
    return _vector(model.get_text_features(**processor(text=[text], return_tensors="pt", padding=True,
                                                         truncation=True))).tolist()


def embed_image(image: Union[str, Path, Image.Image]) -> List[float]:
    if config.EMBEDDING_API_URL and not isinstance(image, Image.Image):
        return _remote({}, image)
    picture = image if isinstance(image, Image.Image) else Image.open(image)
    model, processor = _model()
    return _vector(model.get_image_features(**processor(images=picture.convert("RGB"), return_tensors="pt"))).tolist()


def _remote(form: dict, image_path=None) -> List[float]:
    files = {"image": open(image_path, "rb")} if image_path else None
    try:
        response = requests.post(f"{config.EMBEDDING_API_URL.rstrip('/')}/embed", data=form, files=files, timeout=120)
    finally:
        if files:
            files["image"].close()
    response.raise_for_status()
    body = response.json()
    return body["image_embedding"] if image_path else body["text_embedding"]
