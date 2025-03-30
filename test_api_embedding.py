import requests
import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import random
import time

# Configuration
DATA_DIR = Path("data")
XRAYS_DIR = DATA_DIR / "PetXrays"
PET_PHOTOS_DIR = DATA_DIR / "PetPhotos"
EMBEDDING_API_URL = "http://localhost:8001/embed"

def print_header(text):
    """Print a formatted header for test sections"""
    print("\n" + "="*80)
    print(f"TEST: {text}")
    print("="*80)

def check_api_status():
    """Check if the embedding API is running"""
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            print("✅ Embedding API is running")
            return True
        else:
            print(f"❌ Embedding API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to Embedding API: {e}")
        return False

def get_file_paths(directory, extensions=None):
    """Get all file paths in a directory with optional extension filtering"""
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []
        
    if extensions:
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if any(f.lower().endswith(ext) for ext in extensions)]
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory)]

def get_embedding(text=None, image_path=None):
    """Get embedding from the API for text and/or image"""
    url = EMBEDDING_API_URL
    
    if not text and not image_path:
        print("❌ Error: Please provide either text or an image path")
        return None
        
    files = {}
    data = {"description": text if text else ""}
    
    if image_path:
        try:
            files = {"image": open(image_path, "rb")}
        except Exception as e:
            print(f"❌ Error opening image file {image_path}: {e}")
            return None
    
    try:
        print(f"📤 Sending request to embedding API with text='{text}', image={os.path.basename(image_path) if image_path else 'None'}")
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed_time = time.time() - start_time
        
        if files and "image" in files:
            files["image"].close()
            
        if response.status_code == 200:
            result = response.json()
            # Look for the embedding in different fields based on the API's response format
            embedding = result.get("embedding")
            if embedding is None:
                embedding = result.get("combined_embedding")
            if embedding is None:
                embedding = result.get("text_embedding")
                
            if embedding:
                print(f"✅ Successfully received embedding with {len(embedding)} dimensions in {elapsed_time:.2f} seconds")
                return embedding, elapsed_time
            else:
                print(f"❌ No embedding found in API response: {result}")
                return None, elapsed_time
        else:
            print(f"❌ Error from embedding API: {response.status_code}")
            print(f"Response content: {response.text}")
            return None, elapsed_time
    except Exception as e:
        print(f"❌ Exception when getting embedding: {e}")
        return None, 0

def test_embedding_quality(embedding):
    """Perform basic quality checks on the embedding"""
    # Convert to numpy array if it's a list
    if isinstance(embedding, list):
        embedding = np.array(embedding)
    
    # Check dimensions
    dim = embedding.shape[0]
    print(f"  Embedding dimension: {dim}")
    if dim < 300:  # Most modern embeddings are at least this size
        print("❌ Embedding dimension seems too small")
        return False
    
    # Check for NaN values
    if np.isnan(embedding).any():
        print("❌ Embedding contains NaN values")
        return False
    
    # Check for zero values
    if (embedding == 0).all():
        print("❌ Embedding is all zeros")
        return False
    
    # Check the L2 norm (should be close to 1 if normalized)
    norm = np.linalg.norm(embedding)
    print(f"  Embedding L2 norm: {norm:.4f}")
    if abs(norm - 1.0) > 0.01:
        print("⚠️ Embedding does not appear to be normalized")
    
    # Check for reasonable variance
    var = np.var(embedding)
    print(f"  Embedding variance: {var:.6f}")
    if var < 1e-4:
        print("⚠️ Embedding has very low variance")
    
    return True

def test_text_embeddings():
    """Test API's ability to generate text embeddings"""
    print_header("TEXT EMBEDDINGS")
    
    test_texts = [
        "Dog with a cough and fever",
        "Cat with dental problems",
        "Limping puppy with a possible fracture",
        "Pet with arthritis needs surgery",
        "X-ray showing broken bone"
    ]
    
    success_count = 0
    timings = []
    
    for text in test_texts:
        print(f"\n🔍 Testing embedding for text: '{text}'")
        embedding, elapsed_time = get_embedding(text=text)
        
        if embedding:
            # Add basic quality checks
            if test_embedding_quality(embedding):
                success_count += 1
                timings.append(elapsed_time)
        else:
            print(f"❌ Failed to get embedding for text: '{text}'")
    
    # Print timing statistics
    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"\n⏱️ Average response time: {avg_time:.2f} seconds")
    
    success_rate = (success_count / len(test_texts)) * 100
    print(f"\nText embedding success rate: {success_rate:.1f}% ({success_count}/{len(test_texts)})")
    
    return success_count == len(test_texts)

def test_image_embeddings():
    """Test API's ability to generate image embeddings"""
    print_header("IMAGE EMBEDDINGS")
    
    # Get all pet photos and X-rays
    pet_photos = get_file_paths(PET_PHOTOS_DIR, ["jpg", "jpeg", "png"])
    xray_images = get_file_paths(XRAYS_DIR, ["jpg", "jpeg", "png"])
    
    # Select a few random images to test
    test_images = []
    if pet_photos:
        test_images.extend(random.sample(pet_photos, min(3, len(pet_photos))))
    if xray_images:
        test_images.extend(random.sample(xray_images, min(3, len(xray_images))))
    
    if not test_images:
        print("❌ No image files found for testing")
        return False
    
    success_count = 0
    timings = []
    
    for image_path in test_images:
        print(f"\n🔍 Testing embedding for image: {os.path.basename(image_path)}")
        embedding, elapsed_time = get_embedding(text="", image_path=image_path)
        
        if embedding:
            # Add basic quality checks
            if test_embedding_quality(embedding):
                success_count += 1
                timings.append(elapsed_time)
        else:
            print(f"❌ Failed to get embedding for image: {os.path.basename(image_path)}")
    
    # Print timing statistics
    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"\n⏱️ Average response time: {avg_time:.2f} seconds")
    
    success_rate = (success_count / len(test_images)) * 100
    print(f"\nImage embedding success rate: {success_rate:.1f}% ({success_count}/{len(test_images)})")
    
    return success_count == len(test_images)

def test_multimodal_embeddings():
    """Test API's ability to generate multimodal embeddings (text + image)"""
    print_header("MULTIMODAL EMBEDDINGS")
    
    # Sample test cases with text and images - UPDATED with correct filenames
    test_cases = [
        ("X-ray showing fracture", XRAYS_DIR / "rocky_1.2.410.200028.100.3.20190927.1032580692.2539.1.1.png"),
        ("Dog with dental issues", PET_PHOTOS_DIR / "emma_brooke-cagle-Ntm4C2lCWxQ-unsplash.jpg"),
        ("Labrador with cough", PET_PHOTOS_DIR / "max_taylor-kopel-WX4i1Jq_o0Y-unsplash.jpg")
    ]
    
    # Filter out test cases with missing files
    valid_test_cases = []
    for text, image_path in test_cases:
        if os.path.exists(image_path):
            valid_test_cases.append((text, image_path))
        else:
            print(f"❌ Image file not found: {image_path}")
    
    if not valid_test_cases:
        print("❌ No valid test cases found")
        return False
    
    success_count = 0
    timings = []
    
    for text, image_path in valid_test_cases:
        print(f"\n🔍 Testing multimodal embedding for text: '{text}' with image: {os.path.basename(image_path)}")
        embedding, elapsed_time = get_embedding(text=text, image_path=image_path)
        
        if embedding:
            # Add basic quality checks
            if test_embedding_quality(embedding):
                success_count += 1
                timings.append(elapsed_time)
        else:
            print(f"❌ Failed to get multimodal embedding")
    
    # Print timing statistics
    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"\n⏱️ Average response time: {avg_time:.2f} seconds")
    
    success_rate = (success_count / len(valid_test_cases)) * 100
    print(f"\nMultimodal embedding success rate: {success_rate:.1f}% ({success_count}/{len(valid_test_cases)})")
    
    return success_count == len(valid_test_cases)

def test_embedding_consistency():
    """Test consistency of embeddings for the same input"""
    print_header("EMBEDDING CONSISTENCY")
    
    # Test text consistency
    text = "Dog with cough and fever"
    print(f"\n🔍 Testing consistency for text: '{text}'")
    
    embedding1, _ = get_embedding(text=text)
    embedding2, _ = get_embedding(text=text)
    
    if embedding1 and embedding2:
        # Convert to numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        similarity = dot_product / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        print(f"  Text embedding consistency - cosine similarity: {similarity:.6f}")
        if similarity > 0.99:
            print("✅ Text embeddings are consistent")
            text_consistent = True
        else:
            print("❌ Text embeddings are not consistent!")
            text_consistent = False
    else:
        print("❌ Failed to get text embeddings for consistency test")
        text_consistent = False
    
    # Test image consistency
    pet_photos = get_file_paths(PET_PHOTOS_DIR, ["jpg", "jpeg", "png"])
    if pet_photos:
        image_path = random.choice(pet_photos)
        print(f"\n🔍 Testing consistency for image: {os.path.basename(image_path)}")
        
        embedding1, _ = get_embedding(text="", image_path=image_path)
        embedding2, _ = get_embedding(text="", image_path=image_path)
        
        if embedding1 and embedding2:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            similarity = dot_product / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            print(f"  Image embedding consistency - cosine similarity: {similarity:.6f}")
            if similarity > 0.99:
                print("✅ Image embeddings are consistent")
                image_consistent = True
            else:
                print("❌ Image embeddings are not consistent!")
                image_consistent = False
        else:
            print("❌ Failed to get image embeddings for consistency test")
            image_consistent = False
    else:
        print("❌ No image files found for consistency test")
        image_consistent = False
    
    return text_consistent and image_consistent

def main():
    """Run all embedding API tests and summarize results"""
    print("\n🧪 EMBEDDING API TEST SCRIPT 🧪\n")
    
    # Check API status
    if not check_api_status():
        print("❌ Embedding API is not running. Please start it with `python app.py`")
        sys.exit(1)
    
    # Run tests
    test_results = {
        "Text Embeddings": test_text_embeddings(),
        "Image Embeddings": test_image_embeddings(),
        "Multimodal Embeddings": test_multimodal_embeddings(),
        "Embedding Consistency": test_embedding_consistency()
    }
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results.items():
        result = "✅ PASS" if passed else "❌ FAIL"
        print(f"{result}: {test_name}")
        all_passed = all_passed and passed
        
    print("\nOVERALL RESULT:", "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    
    # Return success/failure
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 