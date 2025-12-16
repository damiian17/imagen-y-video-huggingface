from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
import torch
from diffusers import AutoPipelineForInpainting
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image, ImageOps
import io
import logging
import asyncio
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Models
pipeline_inpainting = None
model_seg = None
processor_seg = None
model_status = "starting" # starting, loading, ready, failed

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

async def load_models_bg():
    global pipeline_inpainting, model_seg, processor_seg, model_status
    device = get_device()
    logger.info(f"Background loading started on {device}...")
    model_status = "loading"
    try:
        # 1. Load Face Parsing Model (CPU/lightweight)
        logger.info("Loading Face Parsing model...")
        seg_id = "jonathandinu/face-parsing"
        processor_seg = AutoImageProcessor.from_pretrained(seg_id)
        model_seg = SegformerForSemanticSegmentation.from_pretrained(seg_id)
        model_seg.to(device)

        # 2. Load SDXL Inpainting Model
        logger.info("Loading SDXL Inpainting model...")
        inpaint_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        
        pipe = await asyncio.to_thread(
            AutoPipelineForInpainting.from_pretrained,
            inpaint_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            variant="fp16" if device != "cpu" else None,
            use_safetensors=True
        )
        
        if device == "cuda":
            pipe.enable_model_cpu_offload() # Efficient memory for T4
        else:
            pipe.to(device)
            
        pipeline_inpainting = pipe
        model_status = "ready"
        logger.info("All models loaded successfully and ready to serve.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        model_status = "failed"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading task without awaiting it
    asyncio.create_task(load_models_bg())
    yield
    global pipeline_inpainting, model_seg
    pipeline_inpainting = None
    model_seg = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dental specific prompts
STYLE_PROMPTS = {
    "hollywood": "perfect white teeth, porcelain veneers, natural bright smile, high quality dental photography",
    "natural": "clean healthy teeth, natural white enamel, perfect alignment, realistic dental photography",
    "alignment": "perfectly aligned teeth, straight smile, orthodontics result, realistic texture"
}

NEGATIVE_PROMPT = "cavities, rotten teeth, yellow teeth, missing teeth, braces, metal, blur, distortion, low quality, cartoon, noise, artifacts"

def generate_mouth_mask(image_pil):
    """
    Generates a binary mask for the mouth area using Face Parsing.
    """
    global model_seg, processor_seg
    device = get_device()
    
    # Preprocess
    inputs = processor_seg(images=image_pil, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model_seg(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

    # Resize to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_pil.size[::-1], # (height, width)
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0] # (height, width)
    
    # Classes for mouth parts in jonathandinu/face-parsing:
    # 10: mouth, 11: u_lip, 12: l_lip
    mouth_mask = (pred_seg == 10) | (pred_seg == 11) | (pred_seg == 12)
    
    mask_np = mouth_mask.cpu().numpy().astype(np.uint8) * 255
    
    # Dilate mask to ensure coverage of edges (important for veneers/implants)
    kernel = np.ones((15, 15), np.uint8) # Slight dilation
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
    
    return Image.fromarray(dilated_mask)

@app.post("/edit-smile")
async def edit_smile(image: UploadFile = File(...), style: str = Form(...)):
    global pipeline_inpainting, model_status
    
    if model_status != "ready":
        if model_status == "failed":
             raise HTTPException(status_code=500, detail="Model failed to load. Check server logs.")
        return JSONResponse(
            status_code=503, 
            content={"detail": "Model is still loading. Please try again in 1-2 minutes."}
        )

    if style not in STYLE_PROMPTS:
        raise HTTPException(status_code=400, detail="Invalid style selected")

    try:
        contents = await image.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Resize to 1024x1024 max for SDXL
        input_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        # Generate Mask
        try:
            logger.info("Generating mouth mask...")
            mask_image = generate_mouth_mask(input_image)
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to detect face/mouth: {str(e)}")

        prompt = STYLE_PROMPTS[style]
        
        logger.info(f"Inpainting image with style: {style}")
        
        # SDXL Inpainting
        with torch.inference_mode():
             output = pipeline_inpainting(
                 prompt=prompt,
                 negative_prompt=NEGATIVE_PROMPT,
                 image=input_image,
                 mask_image=mask_image,
                 num_inference_steps=30,
                 strength=0.99, # High strength to replace teeth content fully
                 guidance_scale=7.5
             )
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
