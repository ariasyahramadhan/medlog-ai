from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
import cv2
import os
from deepface import DeepFace

app = FastAPI()

# Konfigurasi CORS agar React (Vite/CRA) bisa mengakses server ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ────────────────────────────────────────────────────────────────────

class FaceRegisterRequest(BaseModel):
    image_base64: str

class DosenEntry(BaseModel):
    identifier:  str
    face_vector: list   # embedding dari database Laravel

class FaceVerifyRequest(BaseModel):
    image_base64: str               # Foto baru dari webcam
    dosen_list:   list[DosenEntry]  # Semua dosen beserta vectornya

# ── Helper ────────────────────────────────────────────────────────────────────

def decode_image(image_base64: str):
    """Dekode base64 → numpy array (BGR)."""
    _, encoded = image_base64.split(",", 1)
    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def cosine_similarity(a: list, b: list) -> float:
    """Hitung cosine similarity antara dua vector."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

# ── Endpoint 1: Ekstraksi (Registrasi Biometrik) ──────────────────────────────

@app.post("/extract-face")
async def extract_face(data: FaceRegisterRequest):
    try:
        img  = decode_image(data.image_base64)
        objs = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv"
        )

        if not objs:
            return {"success": False, "message": "Wajah tidak terdeteksi"}

        return {
            "success": True,
            "vector":  objs[0]["embedding"]
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

# ── Endpoint 2: Verifikasi (Login Biometrik) ──────────────────────────────────

@app.post("/verify-face")
async def verify_face(data: FaceVerifyRequest):
    try:
        # 1. Dekode & ekstrak embedding wajah dari kamera
        img  = decode_image(data.image_base64)
        objs = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if not objs:
            return {"success": False, "message": "Wajah tidak terdeteksi"}

        current_vector = objs[0]["embedding"]

        # 2. Bandingkan dengan semua dosen, cari skor tertinggi
        best_score      = -1.0
        best_identifier = None

        for dosen in data.dosen_list:
            score = cosine_similarity(current_vector, dosen.face_vector)
            print(f"[verify-face] {dosen.identifier} → score: {score:.4f}")
            if score > best_score:
                best_score      = score
                best_identifier = dosen.identifier

        # 3. Threshold: > 0.40 dianggap orang yang sama untuk Facenet
        THRESHOLD = 0.85
        is_match  = best_score > THRESHOLD

        if is_match:
            return {
                "success":    True,
                "identifier": best_identifier,
                "score":      best_score,
                "message":    "Verifikasi Berhasil"
            }
        else:
            return {
                "success": False,
                "score":   best_score,
                "message": "Wajah tidak cocok dengan data yang terdaftar"
            }

    except Exception as e:
        print(f"[verify-face] Error: {e}")
        raise HTTPException(status_code=500, detail="Gagal memproses biometrik")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)