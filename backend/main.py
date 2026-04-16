from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np

from core.pipeline import run_pipeline
from config import PCB_DATABASE

app = FastAPI()

@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    pcb_type: str = Form(...)
):

    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    result = run_pipeline(image, pcb_type)

    return result
