# main.py
import os
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from neuralic_core import NeuralicBrain

# Create or use existing state file in working dir
STATE_FILE = os.getenv("NEURALIC_STATE_FILE", "neuralic_state.json")
brain = NeuralicBrain(state_file=STATE_FILE)

app = FastAPI(title="Neuralic 3.0 (Biological brain)")

# Allow CORS for all origins (CHANGE for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # For dev/testing. Replace with specific origin when deploying.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

@app.post("/chat")
async def chat(payload: ChatIn):
    msg = payload.message
    reply = brain.chat(msg)
    return {"reply": reply}

class TeachIn(BaseModel):
    input: str
    reply: str

@app.post("/teach")
async def teach(payload: TeachIn):
    res = brain.teach(payload.input, payload.reply)
    return {"status": res}

# File upload: we use UploadFile (multipart) since you said it's ok to include multipart.
# Add "python-multipart" in requirements.txt
@app.post("/learn_file")
async def learn_file(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        text = str(content)
    res = brain.learn_from_text(text)
    return {"status": res}

@app.get("/stats")
async def stats():
    return brain.stats()

@app.post("/save_now")
async def save_now():
    brain.save_state()
    return {"status": "saved"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # On Render set start command: uvicorn main:app --host 0.0.0.0 --port $PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
