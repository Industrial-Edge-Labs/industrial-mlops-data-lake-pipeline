from __future__ import annotations

import argparse
import hashlib
import time
from datetime import datetime, timezone

import requests

from storage import ArtifactStore, load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MLOps anomaly corpus and publish OTA manifests when thresholds are met."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Evaluate the current corpus once and exit immediately.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate the OTA manifest locally without posting it to the fleet manager.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="Polling interval while waiting for the retraining threshold.",
    )
    return parser.parse_args()


def count_anomalies(store: ArtifactStore) -> int:
    return len(store.list_records())


def build_manifest(store: ArtifactStore) -> dict[str, object]:
    records = store.list_records()
    corpus_fingerprint = hashlib.sha256(
        "".join(record.record_id for record in records).encode("utf-8")
    ).hexdigest()
    version = datetime.now(timezone.utc).strftime("vision-stack-%Y%m%d%H%M%S")

    return {
        "latest_firmware": version,
        "download_url": f"{store.config.artifact_base_url}/models/{version}.tar.gz",
        "sha256_hash": corpus_fingerprint,
        "force_restart": False,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "model_version": version,
        "artifact_kind": "vision-model",
        "notes": f"generated from {len(records)} stored anomaly records",
    }


def publish_manifest(store: ArtifactStore, manifest: dict[str, object], dry_run: bool) -> int:
    if dry_run:
        print("[MLOps] Dry-run enabled. Skipping OTA publication.")
        print(f"[MLOps] Manifest preview: {manifest}")
        return 0

    response = requests.post(store.config.fleet_manager_url, json=manifest, timeout=5.0)
    response.raise_for_status()
    print(f"[MLOps] Fleet Manager accepted OTA manifest: {response.text}")
    return 0


def trigger_retraining_workflow(store: ArtifactStore, dry_run: bool) -> int:
    print("[MLOps] Threshold satisfied. Generating deterministic OTA manifest.")
    time.sleep(1)
    manifest = build_manifest(store)
    print(f"[MLOps] Prepared manifest version={manifest['model_version']} hash={manifest['sha256_hash']}")

    try:
        return publish_manifest(store, manifest, dry_run=dry_run)
    except requests.RequestException as exc:
        print(f"[MLOps] Fleet Manager publication failed: {exc}")
        return 1


def run_sweeper(once: bool, dry_run: bool, poll_seconds: float) -> int:
    config = load_runtime_config()
    store = ArtifactStore(config)
    store.ensure_ready()

    print(
        "[MLOps] Sweeper active. "
        f"storage_mode={config.storage_mode} threshold={config.retraining_threshold}"
    )

    while True:
        count = count_anomalies(store)
        print(f"[MLOps] Current anomaly corpus: {count} / {config.retraining_threshold}")

        if count >= config.retraining_threshold:
            return trigger_retraining_workflow(store, dry_run=dry_run)

        if once:
            print("[MLOps] Threshold not met during one-shot evaluation.")
            return 0

        time.sleep(max(poll_seconds, 0.1))


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_sweeper(once=args.once, dry_run=args.dry_run, poll_seconds=args.poll_seconds))
