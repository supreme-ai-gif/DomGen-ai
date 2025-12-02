# main.py
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from neuralic_hybrid import NeuralicHybrid

PORT = int(os.getenv("PORT", 8000))
STATE_FILE = os.getenv("NEURALIC_STATE_FILE", "neuralic_state.json")
brain = NeuralicHybrid(state_file=STATE_FILE)

app = FastAPI(title="Neuralic 2.2 Hybrid")

# allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

@app.post("/chat")
async def chat(payload: ChatIn):
    reply = brain.chat(payload.message)
    return {"reply": reply}

class TeachIn(BaseModel):
    input: str
    reply: str

@app.post("/teach")
async def teach(payload: TeachIn):
    return {"status": brain.teach(payload.input, payload.reply)}

@app.post("/learn_file")
async def learn_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        text = str(content)
    res = brain.learn_file_text(text)
    return {"status": res}

@app.get("/stats")
async def stats():
    return brain.stats()

@app.post("/save_now")
async def save_now():
    brain.save()
    return {"status":"saved"}

# Serve a static simple frontend if requested
from fastapi.responses import FileResponse, HTMLResponse
@app.get("/", response_class=HTMLResponse)
async def homepage():
    idx = os.path.join("static","index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return HTMLResponse("<h3>Neuralic 2.2 Hybrid — server is running.</h3>")

if __name__ == "__main__":
    # dev run
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
