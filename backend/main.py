from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
import torch
from diffusers import QwenImageEditPlusPipeline
from transformers import BitsAndBytesConfig
from PIL import Image, ImageOps
import io
import logging
import asyncio
import os

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

def load_reference_images():
    """Loads reference images into memory."""
    global REFERENCE_IMAGES
    try:
        # Expected paths - ensure these files exist in Docker container
        refs = {
            "hollywood": "assets/ref_hollywood.png",
            "natural": "assets/ref_natural.png",
            "alignment": "assets/ref_alignment.png"
        }
        for style, path in refs.items():
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                REFERENCE_IMAGES[style] = img
                logger.info(f"Loaded reference image for {style}")
            else:
                logger.warning(f"Reference image not found: {path}")
    except Exception as e:
        logger.error(f"Error loading reference images: {e}")

async def load_model_bg():
    global pipeline, model_status
    device = get_device()
    logger.info(f"Background loading started. Target device context: {device}")
    model_status = "loading"
    try:
        # Simplified Quantization approach to avoid type check error
        # Instead of passing the config object mainly, we can try passing the dict or reliance on device_map auto with boolean
        
        # However, to use NF4 specifically, we need the config.
        # If 'quantization_config' param fails, we pass the arguments that transformers.from_pretrained accepts directly
        # because the pipeline forwards kwargs to the components.
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        logger.info("Loading Qwen-Image-Edit-2509 using NF4 quantization (Direct kwargs)...")
        
        # We run this in a thread to not block the event loop
        pipe = await asyncio.to_thread(
            QwenImageEditPlusPipeline.from_pretrained,
            "Qwen/Qwen-Image-Edit-2509",
            quantization_config=quant_config, # Start with this, if it fails then we fallback to kwargs
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        pipeline = pipe
        load_reference_images()
        
        model_status = "ready"
        logger.info("Model loaded successfully and ready to serve.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_status = "failed"

    # NOTE: If the above fails AGAIN with the same error, we will swap to this logic dynamically or user can instruct:
    # pipe = QwenImageEditPlusPipeline.from_pretrained(..., load_in_4bit=True, ...)

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

    if style not in STYLE_TEXT_PROMPTS: # Flexible if image missing
        raise HTTPException(status_code=400, detail="Invalid style")

    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        prompt = STYLE_TEXT_PROMPTS[style]
        
        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "guidance_scale": 4.5,
            "image_guidance_scale": 1.6
        }

        # Handle Reference Image
        if style in REFERENCE_IMAGES:
             inputs["image"] = [input_image, REFERENCE_IMAGES[style]]
        else:
             logger.warn(f"Missing reference image for {style}, using single image input")
             inputs["image"] = [input_image]

        logger.info(f"Processing image with style: {style}")
        
        with torch.inference_mode():
             output = pipeline(**inputs)
             output_image = output.images[0]
        
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health():
    return {"status": model_status, "device": get_device()}
