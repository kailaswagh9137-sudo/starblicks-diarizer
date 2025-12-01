import os
from fastapi import FastAPI, File, UploadFile
from pyannote.audio import Pipeline

HF_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

app = FastAPI()

@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    with open("tmp.wav", "wb") as f:
        f.write(await audio.read())

    diarization = pipeline("tmp.wav")

    results = []
    for turn, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })

    return {"status": "ok", "segments": results}

@app.get("/")
def home():
    return {"service": "STARBLICKS DIARIZER LIVE"}
