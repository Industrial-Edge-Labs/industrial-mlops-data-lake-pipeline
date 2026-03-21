import io
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from minio import Minio
from minio.error import S3Error
import uvicorn

app = FastAPI(title="Industrial Data Lake Ingestion API")

# Configure MinIO Object Store Connection
# In production, these should be bound to secure Vault arguments
MINIO_CLIENT = Minio(
    "127.0.0.1:9000",
    access_key="admin",
    secret_key="IndustrialEdgeLabs2026",
    secure=False
)

BUCKET_NAME = "edge-anomalies"

def ensure_bucket_exists():
    try:
        found = MINIO_CLIENT.bucket_exists(BUCKET_NAME)
        if not found:
            MINIO_CLIENT.make_bucket(BUCKET_NAME)
            print(f"[DataLake] Created Long-Term Storage Bucket: {BUCKET_NAME}")
    except S3Error as e:
        print(f"[DataLake] S3 API Target Unreachable: {e}")

@app.on_event("startup")
async def startup_event():
    ensure_bucket_exists()

@app.post("/api/v1/datalake/upload")
async def upload_anomaly(
    device_id: str = Form(...),
    confidence: float = Form(...),
    anomaly_image: UploadFile = File(...)
):
    """
    Ingests binary defect crops from Edge Detection Modules (#4) alongside 
    structured telemetry metadata asynchronously without blocking the edge runtime.
    """
    if not anomaly_image.content_type.startswith("image/"):
        raise HTTPException(400, "Payload stream must be a valid imaging MIME boundary")

    file_bytes = await anomaly_image.read()
    file_stream = io.BytesIO(file_bytes)
    
    object_name = f"{device_id}/{datetime.utcnow().strftime('%Y-%m-%d')}/{uuid.uuid4()}.jpg"

    try:
        MINIO_CLIENT.put_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
            data=file_stream,
            length=len(file_bytes),
            content_type=anomaly_image.content_type,
            metadata={
                "Device-Id": device_id,
                "Confidence-Score": str(confidence),
                "Ingestion-Time": datetime.utcnow().isoformat()
            }
        )
        return {
            "status": "success",
            "message": "Anomaly snapshot safely persisted to Long-Term Storage",
            "bucket_path": f"s3://{BUCKET_NAME}/{object_name}"
        }
    except S3Error as err:
        raise HTTPException(500, detail=str(err))

if __name__ == "__main__":
    uvicorn.run("ingestion_api:app", host="0.0.0.0", port=8000, reload=True)
