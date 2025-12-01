import os
from fastapi import FastAPI, File, UploadFile
from pyannote.audio import Pipeline
import torch

HF_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

app = FastAPI()

@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    with open("tmp.wav", "wb") as f:
        f.write(await audio.read())

    diarization = pipeline("tmp.wav")

    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker
        })

    return {"status": "ok", "segments": result}

@app.get("/")
def home():
    return {"service": "Starblicks Diarizer Running ✔"}
