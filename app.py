#!/usr/bin/env python3
"""
Embedding API (optional): CLIP embeddings over HTTP, for clients that shouldn't load the model.

    python app.py                      # http://127.0.0.1:8001 (HOST=0.0.0.0 to listen on the network)
    EMBEDDING_API_URL=http://127.0.0.1:8001 python store_with_embedding.py   # use it instead of local CLIP

POST /embed with form fields `text` (or `description`) and/or a file `image`: returns text_embedding,
image_embedding and `embedding` (the image's if there is one, else the text's). Embeddings are
L2-normalized; text and image embeddings are returned separately, not averaged.
"""
import io
import os
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from claims import config
from claims.embeddings import embed_image, embed_text

# This process computes embeddings itself
config.EMBEDDING_API_URL = ""
app = FastAPI(title="Multimodal Claims Embedding API", version="2026.10")


class EmbeddingResponse(BaseModel):
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    embedding: List[float]
    model: str


@app.get("/")
def health():
    return {"status": "ok", "model": config.CLIP_MODEL}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(text: Optional[str] = Form(None), description: Optional[str] = Form(None),
                image: Optional[UploadFile] = File(None)):
    words = text if text is not None else description
    if not words and image is None:
        raise HTTPException(status_code=422, detail="Send `text` (or `description`), an `image`, or both")
    text_vector = embed_text(words) if words else None
    image_vector = None
    if image is not None:
        try:
            image_vector = embed_image(Image.open(io.BytesIO(await image.read())))
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"{image.filename} is not an image")
    return EmbeddingResponse(text_embedding=text_vector, image_embedding=image_vector,
                             embedding=image_vector or text_vector, model=config.CLIP_MODEL)


@app.post("/embed_file", response_model=EmbeddingResponse)
def embed_file(text: Optional[str] = Form(None), description: Optional[str] = Form(None),
               image_path: Optional[str] = Form(None)):
    """Embed an image from the repository's data folder (paths outside it are refused)."""
    words = text if text is not None else description
    image_vector = None
    if image_path:
        path = (config.ROOT / image_path).resolve()
        if not path.is_relative_to(config.DATA.resolve()) or not path.is_file():
            raise HTTPException(status_code=400, detail="image_path must be a file under data/")
        image_vector = embed_image(path)
    if not words and image_vector is None:
        raise HTTPException(status_code=422, detail="Send `text` (or `description`), `image_path`, or both")
    text_vector = embed_text(words) if words else None
    return EmbeddingResponse(text_embedding=text_vector, image_embedding=image_vector,
                             embedding=image_vector or text_vector, model=config.CLIP_MODEL)


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8001")))
