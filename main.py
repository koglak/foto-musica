from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from mido import Message, MidiFile, MidiTrack
import uuid
import cv2
from io import BytesIO

app = FastAPI()

# Kullanıcı seçeneklerine varsayılanlar
INSTRUMENTS = {
    "piano": 1,
    "strings": 49,
    "synth": 81,
    "guitar": 25,
    "bass": 34
}

PITCH_RANGES = {
    "low": (20, 60),
    "mid": (40, 80),
    "high": (60, 100)
}

RESOLUTIONS = {
    "coarse": 16,
    "medium": 32,
    "fine": 64
}


@app.post("/photo-to-music")
async def photo_to_music(
    image: UploadFile = File(...),
    instrument: str = Form("piano"),
    pitch: str = Form("mid"),
    resolution: str = Form("medium"),
    tempo: int = Form(90)
):
    img = Image.open(image.file).convert("L")
    size = RESOLUTIONS.get(resolution, 32)
    img = img.resize((size, size))
    pixels = np.array(img)

    min_pitch, max_pitch = PITCH_RANGES.get(pitch, (40, 80))
    pitch_range = max_pitch - min_pitch
    selected_instrument = INSTRUMENTS.get(instrument, 1)

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change',
                 program=selected_instrument, time=0))

    ticks_per_beat = mid.ticks_per_beat
    tick_duration = int((60 / tempo) * ticks_per_beat * 0.5)

    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            brightness = pixels[y, x]
            note = min_pitch + int((brightness / 255) * pitch_range)
            track.append(Message('note_on', note=note, velocity=64, time=0))
            track.append(Message('note_off', note=note,
                         velocity=64, time=tick_duration))

    midi_bytes = BytesIO()
    mid.save(file=midi_bytes)
    midi_bytes.seek(0)

    return StreamingResponse(midi_bytes, media_type="audio/midi", headers={"Content-Disposition": "attachment; filename=music.mid"})


MOOD_CONFIG = {
    # Piyano
    "sakin":     {"instrument": 0,  "tempo": 70,  "scale": [60, 62, 63, 65, 67, 68, 70]},
    # Synth
    "elektro":   {"instrument": 81, "tempo": 130, "scale": [60, 64, 67, 71, 74]},
    # Gitar
    "gitar":     {"instrument": 25, "tempo": 100, "scale": [60, 62, 64, 65, 67, 69, 71]},
    # Strings
    "kaotik":    {"instrument": 49, "tempo": 150, "scale": [60, 61, 63, 66, 70]}
}


def compute_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.mean(gray)


def compute_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size


def compute_symmetry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flipped = np.fliplr(gray)
    diff = np.abs(gray - flipped)
    return 1 - (np.mean(diff) / 255)


def decide_mood(brightness, edge_density, symmetry):
    if brightness < 100 and symmetry > 0.7:
        return "sakin"
    elif brightness > 150 and edge_density > 0.2 and symmetry < 0.5:
        return "elektro"
    elif 100 <= brightness <= 150:
        return "gitar"
    else:
        return "kaotik"


@app.post("/photo-to-music-rules")
async def photo_to_music_rules(image: UploadFile = File(...)):
    pil_image = Image.open(image.file).convert("RGB")
    img_np = np.array(pil_image)

    brightness = compute_brightness(img_np)
    edge_density = compute_edge_density(img_np)
    symmetry = compute_symmetry(img_np)

    mood = decide_mood(brightness, edge_density, symmetry)

    # MIDI üret ve hafızaya yaz
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    config = MOOD_CONFIG[mood]

    track.append(Message('program_change',
                 program=config["instrument"], time=0))
    ticks_per_beat = mid.ticks_per_beat
    tick_duration = int((60 / config["tempo"]) * ticks_per_beat * 0.25)

    for note in config["scale"]:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note,
                     velocity=64, time=tick_duration))

    midi_bytes = BytesIO()
    mid.save(file=midi_bytes)
    midi_bytes.seek(0)

    return StreamingResponse(
        midi_bytes,
        media_type="audio/midi",
        headers={
            "Content-Disposition": f"attachment; filename={mood}_music.mid",
            "X-Mood": mood,
            "X-Brightness": str(round(brightness, 2)),
            "X-Edge-Density": str(round(edge_density, 3)),
            "X-Symmetry": str(round(symmetry, 3))
        }
    )
