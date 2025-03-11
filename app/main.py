from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predictions
from app.config import logger
from app.config import API_VERSION
from contextlib import asynccontextmanager

app = FastAPI(
    title="Alzheimer's Detection API",
    description="API for detecting Alzheimer's disease stages from MRI scans",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(predictions.router, tags=["Predictions"])


@asynccontextmanager
async def lifespan():
    logger.info("API is starting up...")


@app.get("/")
async def is_running():
    return {"status": "MRI Analysis Service API is Running"}

