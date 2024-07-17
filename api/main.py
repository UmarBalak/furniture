# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def health_check():
#     return "The health check is successful!"

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
from furniture import detect_furniture, get_top_complementary_items

app = FastAPI()

@app.get("/")
async def health_check():
    return "The health check is successful!"

@app.post("/detect")
async def detect_furniture_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only jpg, jpeg, and png are accepted.")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((640, 480))  # Resize to match model's expected input size

    detected_furniture = detect_furniture(image)
    if not detected_furniture:
        return JSONResponse(content={"detail": "No furniture detected in the image."}, status_code=200)

    top_complementary_furniture = get_top_complementary_items(detected_furniture)

    response = {
        "detected_furniture": detected_furniture,
        "top_complementary_furniture": top_complementary_furniture
    }

    return JSONResponse(content=response, status_code=200)

@app.post("/recommend")
async def recommend_furniture_endpoint(detected_furniture: list):
    top_complementary_furniture = get_top_complementary_items(detected_furniture)
    
    response = {
        "top_complementary_furniture": top_complementary_furniture
    }

    return JSONResponse(content=response, status_code=200)
