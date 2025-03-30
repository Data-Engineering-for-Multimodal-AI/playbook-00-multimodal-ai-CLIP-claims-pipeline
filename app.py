from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import os

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    embedding: List[float]  # Unified embedding field for easier use

@app.get("/")
def read_root():
    return {"message": "Multimodal Claims Embedding API is running"}

@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(
    description: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        # Use text if provided, otherwise use description
        input_text = text if text is not None else description
        
        if input_text is None:
            logger.error("No text or description provided")
            raise HTTPException(status_code=422, detail="Either 'text' or 'description' parameter is required")
        
        logger.info(f"Processing text: {input_text[:50]}...")
        
        # Process text
        inputs = processor(text=input_text, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)
        text_embedding = text_features.detach().numpy()[0].tolist()
        
        # Process image if provided
        image_embedding = None
        combined_embedding = None
        
        if image:
            try:
                logger.info(f"Processing image: {image.filename}")
                image_content = await image.read()
                img = Image.open(io.BytesIO(image_content))
                
                # Log image details
                logger.info(f"Image size: {img.size}, mode: {img.mode}")
                
                inputs = processor(images=img, return_tensors="pt")
                image_features = model.get_image_features(**inputs)
                image_embedding = image_features.detach().numpy()[0].tolist()
                
                # Create a combined embedding (simple average)
                combined_array = np.mean([
                    text_features.detach().numpy()[0],
                    image_features.detach().numpy()[0]
                ], axis=0)
                combined_embedding = combined_array.tolist()
                logger.info("Successfully created combined embedding")
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Use combined embedding if available, otherwise use text embedding
        final_embedding = combined_embedding if combined_embedding else text_embedding
        
        return {
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "combined_embedding": combined_embedding,
            "embedding": final_embedding
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/embed_file")
async def embed_from_file(
    description: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    image_path: Optional[str] = Form(None)
):
    try:
        # Use text if provided, otherwise use description
        input_text = text if text is not None else description
        
        if input_text is None:
            logger.error("No text or description provided")
            raise HTTPException(status_code=422, detail="Either 'text' or 'description' parameter is required")
        
        logger.info(f"Processing text: {input_text[:50]}...")
        
        # Process text
        inputs = processor(text=input_text, return_tensors="pt", padding=True, truncation=True)
        text_features = model.get_text_features(**inputs)
        text_embedding = text_features.detach().numpy()[0].tolist()
        
        # Process image if path is provided
        image_embedding = None
        combined_embedding = None
        
        if image_path:
            try:
                logger.info(f"Processing image from path: {image_path}")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                img = Image.open(image_path)
                logger.info(f"Image size: {img.size}, mode: {img.mode}")
                
                inputs = processor(images=img, return_tensors="pt")
                image_features = model.get_image_features(**inputs)
                image_embedding = image_features.detach().numpy()[0].tolist()
                
                # Create a combined embedding (simple average)
                combined_array = np.mean([
                    text_features.detach().numpy()[0],
                    image_features.detach().numpy()[0]
                ], axis=0)
                combined_embedding = combined_array.tolist()
                logger.info("Successfully created combined embedding")
            except Exception as e:
                logger.error(f"Error processing image file: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Error processing image file: {str(e)}")
        
        # Use combined embedding if available, otherwise use text embedding
        final_embedding = combined_embedding if combined_embedding else text_embedding
        
        return {
            "text_embedding": text_embedding,
            "image_embedding": image_embedding,
            "combined_embedding": combined_embedding,
            "embedding": final_embedding
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Embedding API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001) 