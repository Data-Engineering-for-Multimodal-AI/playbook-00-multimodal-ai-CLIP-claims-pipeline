from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import numpy as np
from typing import Optional, List, Dict, Any

app = FastAPI(title="Multimodal Claims Embedding API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

class EmbeddingResponse(BaseModel):
    text_embedding: List[float]
    image_embedding: Optional[List[float]] = None
    combined_embedding: Optional[List[float]] = None

@app.get("/")
def read_root():
    return {"message": "Multimodal Claims Embedding API is running"}

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(
    description: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    # Process text
    inputs = processor(text=description, return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    text_embedding = text_features.detach().numpy()[0].tolist()
    
    # Process image if provided
    image_embedding = None
    combined_embedding = None
    
    if image:
        try:
            image_content = await image.read()
            img = Image.open(io.BytesIO(image_content))
            inputs = processor(images=img, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            image_embedding = image_features.detach().numpy()[0].tolist()
            
            # Create a combined embedding (simple average)
            combined_array = np.mean([
                text_features.detach().numpy()[0],
                image_features.detach().numpy()[0]
            ], axis=0)
            combined_embedding = combined_array.tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    return {
        "text_embedding": text_embedding,
        "image_embedding": image_embedding,
        "combined_embedding": combined_embedding
    }

@app.post("/embed_file")
async def embed_from_file(
    description: str = Form(...),
    image_path: Optional[str] = Form(None)
):
    # Process text
    inputs = processor(text=description, return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    text_embedding = text_features.detach().numpy()[0].tolist()
    
    # Process image if path is provided
    image_embedding = None
    combined_embedding = None
    
    if image_path:
        try:
            img = Image.open(image_path)
            inputs = processor(images=img, return_tensors="pt")
            image_features = model.get_image_features(**inputs)
            image_embedding = image_features.detach().numpy()[0].tolist()
            
            # Create a combined embedding (simple average)
            combined_array = np.mean([
                text_features.detach().numpy()[0],
                image_features.detach().numpy()[0]
            ], axis=0)
            combined_embedding = combined_array.tolist()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image file: {str(e)}")
    
    return {
        "text_embedding": text_embedding,
        "image_embedding": image_embedding,
        "combined_embedding": combined_embedding
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 