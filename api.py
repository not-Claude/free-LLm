import os, torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import TinyTransformer
from tokenizer import CharTokenizer

app=FastAPI(title="WikiTinyLM")
model=None
tok=None
device="cpu"

class Req(BaseModel):
    prompt:str
    max_new_tokens:int=100

@app.on_event("startup")
def load():
    global model,tok,device
    if not os.path.exists("checkpoint.pt") or not os.path.exists("tokenizer.txt"):
        return
    device="cuda" if torch.cuda.is_available() else "cpu"
    tok=CharTokenizer.load("tokenizer.txt")
    model=TinyTransformer(tok.vocab_size,block_size=256).to(device)
    s=torch.load("checkpoint.pt",map_location=device)
    model.load_state_dict(s["model"])
    model.eval()

@app.get("/")
def home():
    return FileResponse("web/index.html")

@app.post("/generate")
def generate(r:Req):
    if model is None:
        return {"error":"Модель ещё не обучена"}
    ids=torch.tensor([tok.encode(r.prompt)],device=device)
    with torch.no_grad():
        for _ in range(min(r.max_new_tokens,256)):
            logits=model(ids[:,-256:])
            probs=torch.softmax(logits[:,-1],dim=-1)
            nxt=torch.multinomial(probs,1)
            ids=torch.cat([ids,nxt],dim=1)
    return {"text":tok.decode(ids[0].tolist())}
