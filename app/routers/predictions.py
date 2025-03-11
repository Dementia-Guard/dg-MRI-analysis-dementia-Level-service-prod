from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.model_loader import load_model
from app.utils.preprocess import preprocess_image
from app.config import CLASS_NAMES, MODEL_PATH, logger
import numpy as np

router = APIRouter()

# Load the model at startup
model = load_model(MODEL_PATH)


# API root to confirm it's running
@router.get("/details")
async def root():
    return {
        "message": "Alzheimer's Detection API is running",
        "supported_classes": CLASS_NAMES,
    }


# Predict Alzheimer's stage from an MRI scan
@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        image_bytes = await file.read()
        img_array, preprocessing_details = preprocess_image(image_bytes)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]

        confidence_scores = {
            class_name: float(score)
            for class_name, score in zip(CLASS_NAMES, predictions[0])
        }

        return {
            "predicted_class": predicted_class,
            "confidence_scores": confidence_scores,
            "preprocessing_details": preprocessing_details,
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
