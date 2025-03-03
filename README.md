# Document Generator API

A FastAPI-based web application that uses Ollama to generate documents through a REST API.

## Requirements

- Docker and Docker Compose
- Ubuntu 24.04 (or other compatible OS)
- Ollama supported hardware

## Setup Instructions

### Option 1: Using Docker Compose (Recommended)

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd document-generator
   ```

2. Start the application with Docker Compose:
   ```bash
   docker-compose up -d
   ```

   This will:
   - Build and start your FastAPI application
   - Start Ollama in a separate container
   - Set up networking between them

3. Ollama will automatically download the default model (llama3) on first use. To pre-download:
   ```bash
   docker exec -it document-generator_ollama_1 ollama pull llama3
   ```

4. Access the API at http://localhost:8000

### Option 2: Manual Setup

1. Install Ollama on Ubuntu 24:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Download the default model:
   ```bash
   ollama pull llama3
   ```

4. In a separate terminal, set up the Python application:
   ```bash
   # Clone repository
   git clone <your-repo-url>
   cd document-generator
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Start the application
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

- `GET /` - Health check endpoint
- `GET /models` - List available Ollama models
- `POST /generate` - Generate a document based on prompt
- `POST /generate/stream` - Generate a document with streaming response

## Example Usage

### Generate Document

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a formal business letter to schedule a meeting",
    "model": "llama3",
    "system_prompt": "You are an expert business writer",
    "format": "markdown"
  }'
```

### Stream Document Generation

```bash
curl -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a detailed project proposal for a web application",
    "model": "llama3"
  }'
```

## Configuration

Environment variables:
- `OLLAMA_API_BASE_URL` - URL for Ollama API (default: http://localhost:11434/api)
- `DEFAULT_OLLAMA_MODEL` - Default model to use (default: llama3)

## License

[Your License]