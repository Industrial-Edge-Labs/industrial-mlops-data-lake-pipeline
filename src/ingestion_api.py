from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import uvicorn

from storage import ArtifactStore, UploadMetadata, load_runtime_config


app = FastAPI(title="Industrial MLOps Data Lake Ingestion API")

RUNTIME_CONFIG = load_runtime_config()
STORE = ArtifactStore(RUNTIME_CONFIG)


@app.on_event("startup")
async def startup_event() -> None:
    STORE.ensure_ready()


@app.get("/healthz")
async def healthz() -> dict[str, object]:
    stats = STORE.stats()
    return {
        "status": "ok",
        "storage_mode": stats["storage_mode"],
        "records_total": stats["records_total"],
        "retraining_threshold": stats["retraining_threshold"],
    }


@app.get("/api/v1/datalake/stats")
async def datalake_stats() -> dict[str, object]:
    return STORE.stats()


@app.post("/api/v1/datalake/upload")
async def upload_anomaly(
    device_id: str = Form(...),
    timestamp_ns: int = Form(...),
    frame_id: int = Form(...),
    confidence: float = Form(...),
    class_id: int = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    width: float = Form(...),
    height: float = Form(...),
    source_repo: str = Form("industrial-visual-inspection-engine"),
    model_version: str = Form(""),
    anomaly_image: UploadFile = File(...),
) -> dict[str, object]:
    content_type = anomaly_image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="anomaly_image must be an image payload")

    file_bytes = await anomaly_image.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="anomaly_image cannot be empty")

    try:
        record = STORE.append_upload(
            UploadMetadata(
                device_id=device_id,
                timestamp_ns=timestamp_ns,
                frame_id=frame_id,
                confidence=confidence,
                class_id=class_id,
                x=x,
                y=y,
                width=width,
                height=height,
                source_repo=source_repo,
                model_version=model_version,
            ),
            file_bytes=file_bytes,
            filename=anomaly_image.filename or "anomaly.bin",
            content_type=content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "success",
        "record_id": record.record_id,
        "storage_mode": record.storage_mode,
        "storage_uri": record.storage_uri,
        "frame_id": record.frame_id,
        "class_id": record.class_id,
    }


if __name__ == "__main__":
    uvicorn.run("ingestion_api:app", host="127.0.0.1", port=8000, reload=False)
