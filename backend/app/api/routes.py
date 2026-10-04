"""HTTP routes. Scoring stays in the loaded prediction service."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.schemas.prediction import SequenceRequest
from backend.app.services.prediction_service import ArtifactError, PredictionService

router = APIRouter()


def _browser_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    if any(origin == "*" for origin in origins):
        raise RuntimeError("CORS_ORIGINS must list explicit origins. A wildcard is not allowed.")
    if not origins:
        raise RuntimeError("CORS_ORIGINS did not contain an origin.")
    return origins


def create_app(manifest_path: Path | str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = PredictionService.load(manifest_path=manifest_path)
        yield

    app = FastAPI(title="Sepsis early-warning research API", version="1", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_browser_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )
    app.include_router(router)

    @app.exception_handler(ValueError)
    def _invalid_input(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": "invalid_input", "detail": str(exc)})

    @app.exception_handler(ArtifactError)
    def _artifact_error(_request: Request, exc: ArtifactError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": str(exc)})

    return app


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/api/model-info")
def model_info(request: Request) -> dict:
    return request.app.state.service.model_info()


@router.post("/api/predict")
def predict_current(body: SequenceRequest, request: Request) -> dict:
    return request.app.state.service.predict_current(_payload(body))


@router.post("/api/predict-sequence")
def predict_sequence(body: SequenceRequest, request: Request) -> dict:
    return request.app.state.service.predict_sequence(_payload(body))


def _payload(body: SequenceRequest) -> dict:
    observations = []
    for item in body.observations:
        record = {"values": item.values}
        if item.hour is not None:
            record["hour"] = item.hour
        observations.append(record)
    return {"observations": observations}
