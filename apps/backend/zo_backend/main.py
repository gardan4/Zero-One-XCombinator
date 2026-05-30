from __future__ import annotations

from zo_common.env import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from zo_backend.api.copilot import router as copilot_router
from zo_backend.api.runs import router as runs_router
from zo_backend.config import settings

app = FastAPI(title="Zero One Control Plane", version="0.1.0")

_origins = (
    ["*"]
    if settings.cors_origins.strip() == "*"
    else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(runs_router, prefix="/api")
app.include_router(copilot_router, prefix="/api")
