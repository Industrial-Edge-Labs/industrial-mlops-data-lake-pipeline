from __future__ import annotations

import json
import mimetypes
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:
    Minio = None
    S3Error = Exception


DEFAULT_STORAGE_MODE = "filesystem"
DEFAULT_RUNTIME_DIR = Path("runtime")
DEFAULT_BUCKET_NAME = "edge-anomalies"
DEFAULT_MINIO_ENDPOINT = "127.0.0.1:9000"
DEFAULT_MINIO_ACCESS_KEY = "admin"
DEFAULT_MINIO_SECRET_KEY = "IndustrialEdgeLabs2026"
DEFAULT_ARTIFACT_BASE_URL = "https://artifacts.industrial-edge.local"
DEFAULT_FLEET_MANAGER_URL = "http://127.0.0.1:8080/api/v1/nodes/ota-manifest"
DEFAULT_RETRAINING_THRESHOLD = 1000
DEFAULT_SOURCE_REPO = "industrial-visual-inspection-engine"

VALID_STORAGE_MODES = frozenset({"filesystem", "minio"})
SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class RuntimeConfig:
    storage_mode: str
    runtime_dir: Path
    index_path: Path
    artifacts_dir: Path
    bucket_name: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_secure: bool
    artifact_base_url: str
    fleet_manager_url: str
    retraining_threshold: int


@dataclass(frozen=True)
class UploadMetadata:
    device_id: str
    timestamp_ns: int
    frame_id: int
    confidence: float
    class_id: int
    x: float
    y: float
    width: float
    height: float
    source_repo: str = DEFAULT_SOURCE_REPO
    model_version: str = ""


@dataclass(frozen=True)
class UploadRecord:
    record_id: str
    storage_mode: str
    storage_uri: str
    object_path: str
    uploaded_at: str
    content_type: str
    file_size_bytes: int
    device_id: str
    timestamp_ns: int
    frame_id: int
    confidence: float
    class_id: int
    x: float
    y: float
    width: float
    height: float
    source_repo: str
    model_version: str


class ArtifactStore:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._client: Optional["Minio"] = None

    def ensure_ready(self) -> None:
        self.config.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.index_path.exists():
            self._persist_records([])

        if self.config.storage_mode == "minio":
            self._client = self._create_minio_client()
            if not self._client.bucket_exists(self.config.bucket_name):
                self._client.make_bucket(self.config.bucket_name)

    def list_records(self) -> list[UploadRecord]:
        if not self.config.index_path.exists():
            return []

        payload = json.loads(self.config.index_path.read_text(encoding="utf-8"))
        return [UploadRecord(**record) for record in payload]

    def append_upload(
        self,
        metadata: UploadMetadata,
        file_bytes: bytes,
        filename: str,
        content_type: str,
    ) -> UploadRecord:
        metadata = normalize_upload_metadata(metadata)
        extension = infer_extension(filename, content_type)
        record_id = str(uuid.uuid4())
        object_path = build_object_path(metadata.device_id, record_id, extension)
        uploaded_at = datetime.now(timezone.utc).isoformat()

        if self.config.storage_mode == "minio":
            storage_uri = self._store_minio(object_path, file_bytes, content_type, metadata)
        else:
            storage_uri = self._store_filesystem(object_path, file_bytes)

        record = UploadRecord(
            record_id=record_id,
            storage_mode=self.config.storage_mode,
            storage_uri=storage_uri,
            object_path=object_path,
            uploaded_at=uploaded_at,
            content_type=content_type,
            file_size_bytes=len(file_bytes),
            device_id=metadata.device_id,
            timestamp_ns=metadata.timestamp_ns,
            frame_id=metadata.frame_id,
            confidence=metadata.confidence,
            class_id=metadata.class_id,
            x=metadata.x,
            y=metadata.y,
            width=metadata.width,
            height=metadata.height,
            source_repo=metadata.source_repo,
            model_version=metadata.model_version,
        )

        records = self.list_records()
        records.append(record)
        records.sort(key=lambda item: (item.timestamp_ns, item.frame_id, item.record_id))
        self._persist_records(records)
        return record

    def stats(self) -> dict[str, object]:
        records = self.list_records()
        latest = asdict(records[-1]) if records else None
        return {
            "storage_mode": self.config.storage_mode,
            "records_total": len(records),
            "bucket_name": self.config.bucket_name if self.config.storage_mode == "minio" else None,
            "index_path": str(self.config.index_path),
            "artifacts_dir": str(self.config.artifacts_dir),
            "latest_record": latest,
            "retraining_threshold": self.config.retraining_threshold,
        }

    def _persist_records(self, records: list[UploadRecord]) -> None:
        payload = [asdict(record) for record in records]
        self.config.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _store_filesystem(self, object_path: str, file_bytes: bytes) -> str:
        target_path = self.config.artifacts_dir / object_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(file_bytes)
        return target_path.resolve().as_uri()

    def _store_minio(
        self,
        object_path: str,
        file_bytes: bytes,
        content_type: str,
        metadata: UploadMetadata,
    ) -> str:
        if self._client is None:
            self._client = self._create_minio_client()

        from io import BytesIO

        self._client.put_object(
            bucket_name=self.config.bucket_name,
            object_name=object_path,
            data=BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
            metadata={
                "device_id": metadata.device_id,
                "frame_id": str(metadata.frame_id),
                "class_id": str(metadata.class_id),
                "confidence": f"{metadata.confidence:.6f}",
                "source_repo": metadata.source_repo,
                "model_version": metadata.model_version,
            },
        )
        return f"s3://{self.config.bucket_name}/{object_path}"

    def _create_minio_client(self) -> "Minio":
        if Minio is None:
            raise RuntimeError("minio package is required when MLOPS_STORAGE_MODE=minio")

        try:
            return Minio(
                self.config.minio_endpoint,
                access_key=self.config.minio_access_key,
                secret_key=self.config.minio_secret_key,
                secure=self.config.minio_secure,
            )
        except S3Error as exc:
            raise RuntimeError(f"unable to initialize MinIO client: {exc}") from exc


def load_runtime_config() -> RuntimeConfig:
    runtime_dir = Path(os.getenv("MLOPS_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR)))
    storage_mode = os.getenv("MLOPS_STORAGE_MODE", DEFAULT_STORAGE_MODE).strip().lower()
    if storage_mode not in VALID_STORAGE_MODES:
        raise ValueError(
            f"MLOPS_STORAGE_MODE must be one of {sorted(VALID_STORAGE_MODES)}, got {storage_mode!r}"
        )

    threshold_raw = os.getenv(
        "MLOPS_RETRAINING_THRESHOLD", str(DEFAULT_RETRAINING_THRESHOLD)
    ).strip()
    try:
        retraining_threshold = int(threshold_raw)
    except ValueError as exc:
        raise ValueError(
            f"MLOPS_RETRAINING_THRESHOLD must be an integer, got {threshold_raw!r}"
        ) from exc
    if retraining_threshold < 0:
        raise ValueError("MLOPS_RETRAINING_THRESHOLD must be zero or positive")

    return RuntimeConfig(
        storage_mode=storage_mode,
        runtime_dir=runtime_dir,
        index_path=runtime_dir / "index.json",
        artifacts_dir=runtime_dir / "artifacts",
        bucket_name=os.getenv("MLOPS_BUCKET_NAME", DEFAULT_BUCKET_NAME).strip(),
        minio_endpoint=os.getenv("MLOPS_MINIO_ENDPOINT", DEFAULT_MINIO_ENDPOINT).strip(),
        minio_access_key=os.getenv("MLOPS_MINIO_ACCESS_KEY", DEFAULT_MINIO_ACCESS_KEY).strip(),
        minio_secret_key=os.getenv("MLOPS_MINIO_SECRET_KEY", DEFAULT_MINIO_SECRET_KEY).strip(),
        minio_secure=os.getenv("MLOPS_MINIO_SECURE", "false").strip().lower()
        in {"1", "true", "yes", "on"},
        artifact_base_url=os.getenv("MLOPS_ARTIFACT_BASE_URL", DEFAULT_ARTIFACT_BASE_URL)
        .strip()
        .rstrip("/"),
        fleet_manager_url=os.getenv("MLOPS_FLEET_MANAGER_URL", DEFAULT_FLEET_MANAGER_URL).strip(),
        retraining_threshold=retraining_threshold,
    )


def normalize_upload_metadata(metadata: UploadMetadata) -> UploadMetadata:
    device_id = sanitize_component(metadata.device_id)
    source_repo = sanitize_component(metadata.source_repo or DEFAULT_SOURCE_REPO)
    model_version = sanitize_component(metadata.model_version) if metadata.model_version else ""

    if metadata.timestamp_ns <= 0:
        raise ValueError("timestamp_ns must be positive")
    if metadata.frame_id <= 0:
        raise ValueError("frame_id must be positive")
    if not 0.0 <= metadata.confidence <= 1.0:
        raise ValueError("confidence must remain within 0.0..1.0")
    if metadata.class_id <= 0:
        raise ValueError("class_id must be positive")
    if metadata.width <= 0.0 or metadata.height <= 0.0:
        raise ValueError("width and height must be positive")

    return UploadMetadata(
        device_id=device_id,
        timestamp_ns=metadata.timestamp_ns,
        frame_id=metadata.frame_id,
        confidence=metadata.confidence,
        class_id=metadata.class_id,
        x=metadata.x,
        y=metadata.y,
        width=metadata.width,
        height=metadata.height,
        source_repo=source_repo,
        model_version=model_version,
    )


def build_object_path(device_id: str, record_id: str, extension: str) -> str:
    day_path = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{sanitize_component(device_id)}/{day_path}/{record_id}{extension}"


def infer_extension(filename: str, content_type: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix:
        return suffix

    guessed = mimetypes.guess_extension(content_type or "")
    if guessed:
        return guessed
    return ".bin"


def sanitize_component(value: str) -> str:
    cleaned = SAFE_COMPONENT_PATTERN.sub("-", value.strip())
    cleaned = cleaned.strip("-.")
    return cleaned or "unknown"
