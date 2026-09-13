from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import json
import os

app = FastAPI(title="PRISM Command Center API")

# Enable CORS for local dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamically resolve the absolute path to the 'data' directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Mount the static data folder
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/")
def read_root():
    return {"status": "PRISM Command Center API is actively running."}

@app.get("/api/telemetry")
def get_telemetry():
    telemetry_path = os.path.join(DATA_DIR, "flight_telemetry.json")
    if os.path.exists(telemetry_path):
        with open(telemetry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    else:
        return JSONResponse(
            content={"error": "Telemetry not found. Run telemetry_parser.py first."}, 
            status_code=404
        )

if __name__ == "__main__":
    import uvicorn
    print(f"Serving data directory from: {DATA_DIR}")
    uvicorn.run(app, host="127.0.0.1", port=8000)