from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image
import io
import logging
import asyncio
import os
import traceback
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Pipeline
pipeline = None
model_status = "starting" # starting, loading, ready, failed

REFERENCE_IMAGES = {}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def download_file(url, dest_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded {dest_path} from {url}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def load_reference_images():
    """Loads reference images into memory. Downloads them if missing."""
    global REFERENCE_IMAGES
    try:
        # Configuration for assets
        assets = {
            "hollywood": {
                "path": "assets/ref_hollywood.png",
                "url": "https://raw.githubusercontent.com/damiian17/frontend-creador-sonrisas/main/public/assets/hollywood.png"
            },
            "natural": {
                "path": "assets/ref_natural.png",
                "url": "https://raw.githubusercontent.com/damiian17/frontend-creador-sonrisas/main/public/assets/natural.png"
            },
            "alignment": {
                "path": "assets/ref_alignment.png",
                "url": "https://raw.githubusercontent.com/damiian17/frontend-creador-sonrisas/main/public/assets/alignment.png"
            }
        }
        
        for style, config in assets.items():
            path = config["path"]
            # Try to download if missing
            if not os.path.exists(path):
                logger.info(f"Asset missing: {path}. Attempting download...")
                download_file(config["url"], path)
            
            # Load if exists (pre-existing or just downloaded)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                REFERENCE_IMAGES[style] = img
                logger.info(f"Loaded reference image for {style}")
            else:
                logger.warning(f"Reference image could not be loaded: {path}")

    except Exception as e:
        logger.error(f"Error loading reference images: {e}")

async def load_model_bg():
    global pipeline, model_status
    device = get_device()
    logger.info(f"Background loading started. Target device context: {device}")
    model_status = "loading"
    try:
        logger.info("Loading Qwen-Image-Edit-2509 in 8-bit (Optimized for A10G, Balanced)...")
        
        pipe = await asyncio.to_thread(
            QwenImageEditPlusPipeline.from_pretrained,
            "Qwen/Qwen-Image-Edit-2509",
            torch_dtype=torch.float16,
            device_map="balanced", 
            load_in_8bit=True
        )
        
        pipeline = pipe
        load_reference_images()
        
        model_status = "ready"
        logger.info("Model loaded successfully and ready to serve.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_status = "failed"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure assets directory exists logic handled by deployment
    asyncio.create_task(load_model_bg())
    yield
    global pipeline
    pipeline = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Text Prompts (Complement visual reference)
STYLE_TEXT_PROMPTS = {
    "hollywood": "Edit the main subject to have the smile and teeth style shown in the reference image. Maintain the subject's identity, skin tone, and facial structure exactly. High quality dental photography, perfect bright white hollywood smile.",
    "natural": "Edit the main subject to have the smile and teeth style shown in the reference image. Maintain the subject's identity, skin tone, and facial structure exactly. High quality dental photography, realistic enamel texture, natural clean healthy smile.",
    "alignment": "Edit the main subject to have the smile and teeth style shown in the reference image. Maintain the subject's identity, skin tone, and facial structure exactly. High quality dental photography, perfect alignment, straight teeth."
}

NEGATIVE_PROMPT = "cartoon, painting, illustration, blur, low quality, bad teeth, missing teeth, extra teeth, fused teeth, yellow teeth, distorted face, changed identity, plastic skin"

@app.post("/edit-smile")
async def edit_smile(image: UploadFile = File(...), style: str = Form(...)):
    global pipeline, model_status
    
    if model_status != "ready":
         if model_status == "failed":
             raise HTTPException(status_code=500, detail="Model failed to load. Check logs.")
         return JSONResponse(status_code=503, content={"detail": "Model is loading. Please wait."})

    if style not in STYLE_TEXT_PROMPTS: 
        raise HTTPException(status_code=400, detail="Invalid style")

    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        prompt = STYLE_TEXT_PROMPTS[style]
        
        inputs = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "guidance_scale": 4.5,
        }

        if style in REFERENCE_IMAGES:
             inputs["image"] = [input_image, REFERENCE_IMAGES[style]]
        else:
             logger.warning(f"WARN: Reference image missing for {style}")
             inputs["image"] = [input_image]

        logger.info(f"Processing image with style: {style}")
        
        with torch.inference_mode():
             output = pipeline(**inputs)
             output_image = output.images[0]
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Error processing image: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)} | Trace: {error_msg}")

@app.get("/")
def health():
    return {"status": model_status, "device": get_device()}
