import os
import math
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from pyannote.audio import Pipeline
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

hf_token = os.getenv("PYANNOTE_AUTH_TOKEN")
if not hf_token:
    raise RuntimeError("ERROR: PYANNOTE_AUTH_TOKEN missing in env vars")

print("Loading diarization model...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

app = FastAPI()

def to_mono16(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path
    ], check=True)

def transcribe_segment(seg_path):
    with open(seg_path, "rb") as f:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="hi",
            temperature=0,
            prompt="Transcribe EXACT speech. DO NOT rewrite."
        )
    return result.text

@app.post("/run")
async def run(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        input_path = tmp.name

    wav_path = input_path + "_16k.wav"
    to_mono16(input_path, wav_path)

    diarization = pipeline(wav_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })
    segments.sort(key=lambda x: x["start"])

    transcript = []

    for seg in segments:
        segfile = wav_path + f"_{seg['start']:.2f}.wav"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", wav_path,
            "-ss", str(seg["start"]),
            "-to", str(seg["end"]),
            segfile
        ], check=True)

        text = transcribe_segment(segfile)
        transcript.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "text": text.strip()
        })
        os.remove(segfile)

    os.remove(input_path)
    os.remove(wav_path)

    return JSONResponse({"status": "ok", "result": transcript})
