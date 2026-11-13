import os
import cv2
import requests
import json
import time
import argparse
from datetime import datetime
from PIL import Image
import numpy as np

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = {
    "image_analysis": "qwen3-vl:32b",
    "text_generation": "gpt-oss:120b",
    "embedding": "nomic-embed-text"
}

def set_ollama_config():
    """Set Ollama environment variables for multi-model support"""
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "3"
    os.environ["OLLAMA_KEEP_ALIVE"] = "-1"

def analyze_image_with_ollama(image_path, model="qwen3-vl:32b"):
    """Send image to Ollama for analysis using Qwen3-VL"""
    try:
        # Convert image to bytes
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_bytes = np.array(image)
        
        # Prepare payload
        payload = {
            "model": model,
            "prompt": "Analyze this bourbon bottle image. Identify: bottle name, distillery, release info, and tasting notes",
            "images": [image_path],
            "options": {
                "temperature": 0.2,
                "num_predict": 2000
            }
        }
        
        # Send to Ollama API
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        
        # Stream response
        result = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                result += data.get("response", "")
                
        return result.strip()
        
    except Exception as e:
        print(f"Image analysis error: {str(e)}")
        return None

def generate_metadata_with_ollama(prompt, model="gpt-oss:120b"):
    """Generate metadata using GPT-OSS in JSON mode"""
    try:
        payload = {
            "model": model,
            "prompt": f"Generate JSON metadata for this bourbon bottle:\n{prompt}\n\nFormat: {json.dumps({'bottle': '', 'distillery': '', 'msrp': '', 'release_info': '', 'tasting_notes': ''})}",
            "options": {
                "temperature": 0.1,
                "num_predict": 1000,
                "json": True
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        result = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                result += data.get("response", "")
                
        return json.loads(result)
        
    except Exception as e:
        print(f"Metadata generation error: {str(e)}")
        return {}

def save_metadata(bottle_name, metadata, raw_analysis):
    """Save information to markdown file"""
    filename = f"bottle_metadata/{bottle_name.lower().replace(' ', '_')}_{datetime.now().timestamp()}.md"
    
    with open(filename, 'w') as f:
        f.write(f"# {bottle_name}\n\n")
        f.write("## Raw Image Analysis\n```\n")
        f.write(raw_analysis)
        f.write("\n```\n\n")
        
        f.write("## Extracted Metadata\n\n")
        for key, value in metadata.items():
            f.write(f"**{key.replace('_', ' ').title()}:** {value}\n\n")
    
    return filename

def main():
    parser = argparse.ArgumentParser(description='Process bourbon bottle images with Ollama')
    parser.add_argument('--image-folder', default='.', help='Folder containing bottle images (default: current directory)')
    parser.add_argument('--model', default='qwen3-vl:32b', help='Ollama model for image analysis')
    args = parser.parse_args()

    set_ollama_config()

    # Ensure output directory exists
    os.makedirs("bottle_metadata", exist_ok=True)

    for filename in os.listdir(args.image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.image_folder, filename)
            
            try:
                print(f"Processing {filename}...")
                
                # Step 1: Analyze image with Qwen3-VL
                raw_analysis = analyze_image_with_ollama(image_path, model=args.model)
                if not raw_analysis:
                    continue
                    
                print(f"\nImage analysis:\n{raw_analysis}")
                
                # Step 2: Generate metadata with GPT-OSS
                metadata = generate_metadata_with_ollama(raw_analysis)
                print(f"\nGenerated metadata: {json.dumps(metadata, indent=2)}")
                
                # Step 3: Save to file
                bottle_name = metadata.get("bottle", "Unknown_Bottle")
                output_path = save_metadata(bottle_name, metadata, raw_analysis)
                print(f"Saved metadata to: {output_path}")
                
                # Respect rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
