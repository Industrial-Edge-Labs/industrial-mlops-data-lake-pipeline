# Industrial MLOps Data Lake Pipeline

This repository ingests inspection-side anomaly evidence, persists the corpus for retraining, and publishes canonical OTA manifests to [Edge Device Fleet Manager](https://github.com/Industrial-Edge-Labs/edge-device-fleet-manager) once retraining thresholds are satisfied.

## Role In The System

- Accepts image uploads aligned with the `InspectionAnomalyPayload` emitted by [Industrial Visual Inspection Engine](https://github.com/Industrial-Edge-Labs/industrial-visual-inspection-engine).
- Stores anomaly evidence in a portable filesystem mode by default, with optional MinIO-backed object storage.
- Generates OTA manifests for [Edge Device Fleet Manager](https://github.com/Industrial-Edge-Labs/edge-device-fleet-manager).

## Routes

- `GET /healthz`
- `GET /api/v1/datalake/stats`
- `POST /api/v1/datalake/upload`

## Runtime

```bash
python src/ingestion_api.py
```

```bash
python src/retraining_trigger.py --once --dry-run
```

Environment variables:

- `MLOPS_STORAGE_MODE` default: `filesystem`
- `MLOPS_RUNTIME_DIR` default: `runtime`
- `MLOPS_BUCKET_NAME` default: `edge-anomalies`
- `MLOPS_MINIO_ENDPOINT` default: `127.0.0.1:9000`
- `MLOPS_MINIO_ACCESS_KEY` default: `admin`
- `MLOPS_MINIO_SECRET_KEY` default: `IndustrialEdgeLabs2026`
- `MLOPS_MINIO_SECURE` default: `false`
- `MLOPS_ARTIFACT_BASE_URL` default: `https://artifacts.industrial-edge.local`
- `MLOPS_FLEET_MANAGER_URL` default: `http://127.0.0.1:8080/api/v1/nodes/ota-manifest`
- `MLOPS_RETRAINING_THRESHOLD` default: `1000`

## Notes

- The ingestion API persists structured metadata alongside image artifacts.
- Filesystem mode keeps the node portable for local validation and CI smoke tests.
- MinIO mode remains available for S3-compatible object storage.
- The external documentation for this node lives in [docs-Industrial-Edge-Labs/industrial-mlops-data-lake-pipeline](https://github.com/Industrial-Edge-Labs/docs-Industrial-Edge-Labs/tree/main/industrial-mlops-data-lake-pipeline).
