import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def load_model(MODEL_PATH):
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Check the path and filename.")
        raise
    except ValueError as e:
        logger.error(f"Invalid model file at {MODEL_PATH}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        raise
