"""Minimal Archiq AI API for testing."""
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok", "service": "archiq-ai-api"}

@app.get("/")
def root():
    return {"message": "Archiq AI API - coming soon"}
