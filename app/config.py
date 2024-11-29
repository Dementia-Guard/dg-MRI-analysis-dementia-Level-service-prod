import logging

# Constants
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
TARGET_SIZE = (64, 64)
MODEL_PATH = "app/models/CNN.keras"
API_VERSION = "v1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
