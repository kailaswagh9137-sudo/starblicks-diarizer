import os
from fastapi import FastAPI, File, UploadFile
from pyannote.audio.pipelines import SpeakerDiarization

HF_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN")

pipeline = SpeakerDiarization.from_pretrained(
    "pyannote/speaker-diarization@3.1",
    use_auth_token=HF_TOKEN
)

app = FastAPI()

@app.post("/diarize")
async def diarize(audio: UploadFile = File(...)):
    with open("tmp.wav", "wb") as f:
        f.write(await audio.read())

    diarization = pipeline("tmp.wav")

    segments = []
    for turn, spk in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": spk,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })

    return {"status": "ok", "segments": segments}

@app.get("/")
def home():
    return {"service": "STARBLICKS DIARIZER LIVE OK"}
