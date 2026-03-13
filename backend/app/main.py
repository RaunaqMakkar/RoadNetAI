"""Third Eye backend API application entrypoint."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import detect, tickets, stats, map, inspection

app = FastAPI(title="Third Eye API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(tickets.router)
app.include_router(stats.router)
app.include_router(map.router)
app.include_router(inspection.router)

# Serve annotated detection frames as static files
FRAMES_DIR = Path(__file__).resolve().parents[1] / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


@app.get("/")
async def root():
    return {"message": "Third Eye Backend Running"}

