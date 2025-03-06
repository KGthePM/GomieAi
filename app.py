# app.py
import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Generator API",
    description="API for generating documents using Ollama models",
    version="1.0.0"
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://10.1.10.144:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama API settings
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "mistral:latest")

# Custom exceptions
class OllamaUnavailableError(Exception):
    """Raised when Ollama service is unavailable"""
    pass

class ModelNotFoundError(Exception):
    """Raised when requested model is not found"""
    pass

class ModelLoadingError(Exception):
    """Raised when there's an error loading the model"""
    pass

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

# Exception handlers
@app.exception_handler(OllamaUnavailableError)
async def ollama_unavailable_exception_handler(request: Request, exc: OllamaUnavailableError):
    logger.error(f"Ollama service unavailable: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Ollama service is unavailable. Please ensure Ollama is running: {str(exc)}"}
    )

@app.exception_handler(ModelNotFoundError)
async def model_not_found_exception_handler(request: Request, exc: ModelNotFoundError):
    logger.error(f"Model not found: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Model not found: {str(exc)}"}
    )

@app.exception_handler(ModelLoadingError)
async def model_loading_exception_handler(request: Request, exc: ModelLoadingError):
    logger.error(f"Error loading model: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Error loading model: {str(exc)}"}
    )

# Utility functions
async def check_ollama_health():
    """Check if Ollama service is responsive"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_API_BASE_URL}/tags")
            if response.status_code != 200:
                return False
            return True
    except Exception:
        return False

async def check_model_exists(model_name: str):
    """Check if a specific model exists in Ollama"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_API_BASE_URL}/tags")
            if response.status_code != 200:
                raise OllamaUnavailableError("Failed to fetch models from Ollama")
            
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            
            return model_name in model_names
    except httpx.RequestError as e:
        raise OllamaUnavailableError(f"Connection error: {str(e)}")

# Static files setup
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

def setup_static_files():
    """Create frontend HTML file if it doesn't exist"""
    index_path = STATIC_DIR / "index.html"
    
    # Only write the file if it doesn't exist
    if not index_path.exists():
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Generator</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .container {
            display: flex;
            gap: 20px;
        }
        .input-area, .output-area {
            flex: 1;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .input-area {
            background-color: #f9f9f9;
        }
        .output-area {
            background-color: #fff;
        }
        textarea, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-family: inherit;
            margin-bottom: 15px;
        }
        textarea {
            min-height: 200px;
            resize: vertical;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        button:disabled {
            background-color: #95a5a6;
            cursor: not-allowed;
        }
        #output {
            white-space: pre-wrap;
            min-height: 300px;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 4px;
            background-color: #fafafa;
            overflow-y: auto;
            font-family: 'Courier New', Courier, monospace;
        }
        .status {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(0, 0, 0, 0.1);
            border-radius: 50%;
            border-top-color: #3498db;
            animation: spin 1s ease-in-out infinite;
            margin-left: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <h1>Document Generator</h1>
    
    <div class="container">
        <div class="input-area">
            <label for="model-select">Model:</label>
            <select id="model-select">
                <option value="mistral:latest">mistral:latest</option>
                <!-- Other models will be populated dynamically -->
            </select>
            
            <label for="system-prompt">System Prompt:</label>
            <textarea id="system-prompt">You are a helpful document generation assistant. Create professional-quality documents based on user requirements.</textarea>
            
            <label for="user-prompt">Your Request:</label>
            <textarea id="user-prompt" placeholder="Describe the document you want to generate...">Write a formal business letter requesting a meeting with a potential client.</textarea>
            
            <div>
                <button id="generate-btn">Generate Document</button>
                <button id="stream-btn">Stream Document</button>
                <span id="loading-indicator" class="loading" style="display: none;"></span>
            </div>
            <p class="status" id="status"></p>
        </div>
        
        <div class="output-area">
            <label for="output">Generated Document:</label>
            <div id="output"></div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const API_BASE_URL = window.location.origin;
            const modelSelect = document.getElementById('model-select');
            const systemPrompt = document.getElementById('system-prompt');
            const userPrompt = document.getElementById('user-prompt');
            const generateBtn = document.getElementById('generate-btn');
            const streamBtn = document.getElementById('stream-btn');
            const output = document.getElementById('output');
            const status = document.getElementById('status');
            const loadingIndicator = document.getElementById('loading-indicator');
            
            // Fetch available models
            async function fetchModels() {
                try {
                    const response = await fetch(`${API_BASE_URL}/models`);
                    if (!response.ok) throw new Error('Failed to fetch models');
                    
                    const data = await response.json();
                    const models = data.models || [];
                    
                    // Clear existing options except the default
                    while (modelSelect.options.length > 1) {
                        modelSelect.remove(1);
                    }
                    
                    // Add models from the API
                    models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.name;
                        option.textContent = model.name;
                        modelSelect.appendChild(option);
                    });
                } catch (error) {
                    console.error('Error fetching models:', error);
                    status.textContent = 'Could not load models. Is the API server running?';
                }
            }
            
            // Generate document (non-streaming)
            async function generateDocument() {
                setLoading(true);
                output.textContent = '';
                status.textContent = 'Generating document...';
                
                try {
                    const response = await fetch(`${API_BASE_URL}/generate`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            model: modelSelect.value,
                            system_prompt: systemPrompt.value,
                            prompt: userPrompt.value,
                            format: 'markdown'
                        })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`API error: ${errorText}`);
                    }
                    
                    const data = await response.json();
                    output.textContent = data.content;
                    status.textContent = 'Document generated successfully!';
                } catch (error) {
                    console.error('Error generating document:', error);
                    status.textContent = `Error: ${error.message}`;
                    output.textContent = 'Failed to generate document. Please try again.';
                } finally {
                    setLoading(false);
                }
            }
            
            // Generate document with streaming
            function streamDocument() {
                setLoading(true);
                output.textContent = '';
                status.textContent = 'Streaming document...';
                
                // Create request data
                const requestData = {
                    model: modelSelect.value,
                    system_prompt: systemPrompt.value,
                    prompt: userPrompt.value
                };
                
                // Use fetch for the initial request
                fetch(`${API_BASE_URL}/generate/stream`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestData)
                }).then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    
                    // Create a new EventSource for the stream
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    
                    function processStream() {
                        reader.read().then(({ done, value }) => {
                            if (done) {
                                status.textContent = 'Document generation complete!';
                                setLoading(false);
                                return;
                            }
                            
                            const chunk = decoder.decode(value, { stream: true });
                            const lines = chunk.split('\n\n');
                            
                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    try {
                                        const data = JSON.parse(line.substring(6));
                                        
                                        if (data.error) {
                                            status.textContent = `Error: ${data.error}`;
                                            setLoading(false);
                                            return;
                                        }
                                        
                                        if (data.content) {
                                            output.textContent += data.content;
                                        }
                                        
                                        if (data.done) {
                                            status.textContent = 'Document generation complete!';
                                            setLoading(false);
                                            return;
                                        }
                                    } catch (e) {
                                        console.error('Error parsing SSE data:', e);
                                    }
                                }
                            }
                            
                            // Continue reading the stream
                            processStream();
                        }).catch(error => {
                            console.error('Stream error:', error);
                            status.textContent = `Error: ${error.message}`;
                            setLoading(false);
                        });
                    }
                    
                    processStream();
                }).catch(error => {
                    console.error('Fetch error:', error);
                    status.textContent = `Error: ${error.message}`;
                    setLoading(false);
                });
            }
            
            function setLoading(isLoading) {
                generateBtn.disabled = isLoading;
                streamBtn.disabled = isLoading;
                loadingIndicator.style.display = isLoading ? 'inline-block' : 'none';
            }
            
            // Event listeners
            generateBtn.addEventListener('click', generateDocument);
            streamBtn.addEventListener('click', streamDocument);
            
            // Initial load
            fetchModels();
        });
    </script>
</body>
</html>"""
        with open(index_path, "w") as f:
            f.write(html_content)

# Call the setup function
setup_static_files()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    ollama_available = await check_ollama_health()
    
    return {
        "status": "online",
        "message": "Document Generator API is running",
        "ollama_status": "available" if ollama_available else "unavailable"
    }

@app.get("/ui", include_in_schema=False)
async def ui_redirect():
    """Redirect to the frontend UI"""
    return RedirectResponse(url="/static/index.html")

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    try:
        ollama_available = await check_ollama_health()
        if not ollama_available:
            raise OllamaUnavailableError("Ollama service is not responding")
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{OLLAMA_API_BASE_URL}/tags")
            if response.status_code != 200:
                raise OllamaUnavailableError(f"Failed to fetch models: HTTP {response.status_code}")
            
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error communicating with Ollama: {str(e)}")
        raise OllamaUnavailableError(f"Connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.post("/generate", response_model=DocumentResponse)
async def generate_document(request: DocumentRequest):
    """Generate a document based on the provided prompt"""
    try:
        # Check if Ollama is available
        ollama_available = await check_ollama_health()
        if not ollama_available:
            raise OllamaUnavailableError("Ollama service is not responding")
        
        # Check if the requested model exists
        model_exists = await check_model_exists(request.model)
        if not model_exists:
            raise ModelNotFoundError(f"Model '{request.model}' not found. Please check available models with GET /models")
        
        # Prepare the request to Ollama
        ollama_request = {
            "model": request.model,
            "prompt": request.prompt,
            "system": request.system_prompt,
            "options": request.options or {}
            "stream": False  # Explicitly disable streaming
        }
        
        # Add retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Send request to Ollama
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{OLLAMA_API_BASE_URL}/generate", 
                        json=ollama_request
                    )
                    
                    if response.status_code == 404:
                        raise ModelNotFoundError(f"Model '{request.model}' not found")
                    
                    if response.status_code != 200:
                        error_text = await response.text()
                        if "loading" in error_text.lower():
                            # Model is loading, retry after delay
                            retry_count += 1
                            logger.info(f"Model '{request.model}' is loading. Retrying in 2 seconds...")
                            await asyncio.sleep(2)
                            continue
                        else:
                            raise HTTPException(
                                status_code=response.status_code, 
                                detail=f"Ollama API error: {error_text}"
                            )
                    
                    result = response.json()
                    logger.info(f"Raw Ollama response: {json.dumps(result)}")

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
                retry_count += 1
                if retry_count >= max_retries:
                    raise HTTPException(status_code=504, detail="Request to Ollama timed out after multiple attempts")
                logger.warning(f"Request timed out, retrying ({retry_count}/{max_retries})...")
                await asyncio.sleep(1)
            
            except (ModelNotFoundError, HTTPException) as e:
                # Don't retry for these exceptions
                raise e
    
    except OllamaUnavailableError as e:
        logger.error(f"Ollama service unavailable: {str(e)}")
        raise
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        raise OllamaUnavailableError(f"Connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")

@app.post("/generate/stream")
async def generate_document_stream(request: Request):
    """Generate a document with streaming"""
    try:
        # Parse request body
        body = await request.json()
        
        # Extract request parameters
        model = body.get("model", DEFAULT_MODEL)
        prompt = body.get("prompt")
        system_prompt = body.get("system_prompt", "You are a helpful document generation assistant.")
        options = body.get("options", {})
        
        # Validate required parameters
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Check if the requested model exists
        model_exists = await check_model_exists(model)
        if not model_exists:
            raise ModelNotFoundError(f"Model '{model}' not found. Please check available models with GET /models")

        # Prepare request to Ollama
        ollama_request = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "options": options,
            "stream": True  # Enable streaming
        }

        async def event_generator():
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    async with client.stream(
                        "POST", 
                        f"{OLLAMA_API_BASE_URL}/generate", 
                        json=ollama_request,
                        timeout=300.0
                    ) as response:
                        if response.status_code != 200:
                            error_detail = await response.text()
                            yield f"data: {json.dumps({'error': f'Ollama API error: {error_detail}'})}\n\n"
                            return
                        
                        async for chunk in response.aiter_lines():
                            if chunk:
                                try:
                                    data = json.loads(chunk)
                                    # Send the chunk in SSE format
                                    yield f"data: {json.dumps({'content': data.get('response', ''), 'done': data.get('done', False)})}\n\n"
                                    
                                    # If this is the final chunk, break
                                    if data.get('done', False):
                                        break
                                except json.JSONDecodeError:
                                    yield f"data: {json.dumps({'error': 'Failed to parse Ollama response'})}\n\n"
                except httpx.TimeoutException:
                    yield f"data: {json.dumps({'error': 'Request to Ollama timed out'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': f'Error: {str(e)}'})}\n\n"
                                    
            # End the stream
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    except OllamaUnavailableError as e:
        logger.error(f"Ollama service unavailable: {str(e)}")
        raise
    except ModelNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error generating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating document: {str(e)}")

# Run the app (for development)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)