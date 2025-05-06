from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mangum import Mangum
from pathlib import Path
import json


app = FastAPI(title="El Camino API", version="0.1.0")


#CORS configuration
# This is needed to allow the frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_PATH = Path(__file__).parent.parent.parent / "frontend" / "static"
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

# Define the path to the templates folder
TEMPLATES_PATH = Path(__file__).parent.parent.parent / "frontend" / "templates"
print(f'Templates path: {TEMPLATES_PATH}')
print(f'Index file path: {TEMPLATES_PATH / "index.html"}')

# Define a Pydantic model for the chatbot request
class ChatbotRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the index.html file."""
    index_file = TEMPLATES_PATH / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="index.html not found", status_code=404)

@app.post("/chatbot")
async def handle_prompt(request: ChatbotRequest):
    """Handle chatbot prompts."""
    input_text = request.prompt

    # Create conversation history string
    history = "\n".join(conversation_history)

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history, input_text, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length=60)

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(response)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

