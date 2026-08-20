import os
import gc
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama

MODEL_PATH = "models/model.gguf"
MAX_RAM_GB = 5.0
CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)
CONTEXT_SIZE = 2048

def ram_available_gb():
    return psutil.virtual_memory().available / (1024 ** 3)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Модель не найдена: {MODEL_PATH}")

if ram_available_gb() < 1.0:
    raise RuntimeError("Недостаточно свободной оперативной памяти.")

llm = Llama(
    model_path=MODEL_PATH,
    n_threads=CPU_THREADS,
    n_ctx=CONTEXT_SIZE,
    n_batch=64,
    n_gpu_layers=0,
    verbose=False,
)

gc.collect()

app = FastAPI(title="Free LLM API", version="1.0.0")
app.mount("/static", StaticFiles(directory="web"), name="static")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "local-llm"
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7

@app.get("/")
def home():
    return FileResponse("web/index.html")

@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{
            "id": "local-llm",
            "object": "model",
            "owned_by": "free-llm"
        }]
    }

@app.get("/health")
def health():
    process = psutil.Process(os.getpid())
    return {
        "status": "ok",
        "ram_available_gb": round(ram_available_gb(), 2),
        "ram_used_by_server_gb": round(
            process.memory_info().rss / (1024 ** 3), 2
        )
    }

@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is empty")

    prompt = ""
    for message in request.messages:
        if message.role == "system":
            prompt += f"System: {message.content}\n"
        elif message.role == "user":
            prompt += f"User: {message.content}\n"
        elif message.role == "assistant":
            prompt += f"Assistant: {message.content}\n"

    prompt += "Assistant:"
    max_tokens = min(max(1, request.max_tokens), 512)

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=0.9,
        stop=["User:", "\nUser:"],
    )

    text = result["choices"][0]["text"].strip()

    return {
        "id": "chat-local",
        "object": "chat.completion",
        "model": "local-llm",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }]
    }
