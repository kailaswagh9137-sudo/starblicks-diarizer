import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pyannote.audio import Pipeline

HF_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("ERROR: PYANNOTE_AUTH_TOKEN missing in environment variables")

print("Loading pyannote model...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

app = FastAPI()

@app.get("/")
def home():
    return {"service": "STARBLICKS DIARIZER READY ✔"}

@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    tmp = "input.wav"
    with open(tmp, "wb") as f:
        f.write(await audio.read())

    diarization = pipeline(tmp)

    segments = []
    for turn, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })

    os.remove(tmp)

    return JSONResponse({"status": "ok", "segments": segments})
