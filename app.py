# app.py
import os
from typing import Dict, Any, Optional
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Document Generator API",
    description="API for generating documents using Ollama models",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama API settings
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "llama3")

# Pydantic models for request/response validation
class DocumentRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_MODEL
    system_prompt: Optional[str] = "You are a helpful document generation assistant."
    format: Optional[str] = "markdown"
    options: Optional[Dict[str, Any]] = None

class DocumentResponse(BaseModel):
    content: str
    model: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Document Generator API is running"}

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{OLLAMA_API_BASE_URL}/tags")
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch models from Ollama")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {str(e)}")

@app.post("/generate", response_model=DocumentResponse)
async def generate_document(request: DocumentRequest):
    """Generate a document based on the provided prompt"""
    try:
        # Prepare the request to Ollama
        ollama_request = {
            "model": request.model,
            "prompt": request.prompt,
            "system": request.system_prompt,
            "options": request.options or {}
        }
        
        # Send request to Ollama
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_API_BASE_URL}/generate", 
                json=ollama_request
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"Ollama API error: {response.text}"
                )
            
            result = response.json()
            
            # Format output based on requested format
            if request.format == "markdown":
                # Ollama returns already formatted text
                content = result.get("response", "")
            else:
                # Handle other formats if needed
                content = result.get("response", "")
            
            return DocumentResponse(
                content=content,
                model=request.model
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to Ollama timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")

# Run the app (for development)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)