from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from contextlib import asynccontextmanager
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageOps
import io
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = None
model_status = "starting" # starting, loading, ready, failed

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

async def load_model_bg():
    global pipeline, model_status
    device = get_device()
    logger.info(f"Background loading started on {device}...")
    model_status = "loading"
    try:
        # Load InstructPix2Pix
        model_id = "timbrooks/instruct-pix2pix"
        
        # Run synchronous loading in a separate thread
        pipe = await asyncio.to_thread(
            StableDiffusionInstructPix2PixPipeline.from_pretrained,
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            safety_checker=None 
        )
        
        pipe.to(device)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        pipeline = pipe
        model_status = "ready"
        logger.info(f"Model {model_id} loaded successfully and ready to serve.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_status = "failed"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading task without awaiting it
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

STYLE_PROMPTS = {
    "hollywood": "make his/her teeth look like a perfect bright white hollywood smile",
    "natural": "make the teeth look clean, healthy and natural white",
    "alignment": "make the teeth perfectly aligned and straight"
}

@app.post("/edit-smile")
async def edit_smile(image: UploadFile = File(...), style: str = Form(...)):
    global pipeline, model_status
    
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
        
        # Resize if too large to ensure speed/memory safety
        input_image = ImageOps.contain(input_image, (1024, 1024))

        prompt = STYLE_PROMPTS[style]
        
        inputs = {
            "prompt": prompt,
            "image": input_image,
            "num_inference_steps": 20, 
            "image_guidance_scale": 1.5,
        }
        
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
