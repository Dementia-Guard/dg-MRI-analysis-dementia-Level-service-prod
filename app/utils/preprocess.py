import numpy as np
from PIL import Image
from fastapi import HTTPException
from app.config import TARGET_SIZE, logger
import io


def preprocess_image(image_bytes: bytes):
    """
    Preprocess the uploaded image for prediction.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))

        preprocessing_details = {
            "original_size": img.size,
            "original_mode": img.mode,
            "target_size": TARGET_SIZE,
        }

        img = img.convert("L")
        img = img.resize(TARGET_SIZE)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(-1, TARGET_SIZE[0], TARGET_SIZE[1], 1)

        return img_array, preprocessing_details
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=422, detail="Unable to process image")
