"""
Pipeline & training control endpoints for ThermoSense API.

POST /pipeline/backfill   - Run the data backfill pipeline
POST /pipeline/daily      - Run the daily incremental update
POST /pipeline/train      - Train models
GET  /pipeline/status     - Get system status (data, models, etc.)
GET  /pipeline/logs       - Stream recent log output
GET  /pipeline/mlflow     - Get MLflow experiment runs
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_PATH = _PROJECT_ROOT / "data" / "processed" / "daily_merged.parquet"
FEATURES_PATH = _PROJECT_ROOT / "data" / "features" / "feature_matrix.parquet"
MODELS_DIR = _PROJECT_ROOT / "models"
RESULTS_PATH = MODELS_DIR / "results.json"

_job_store: Dict[str, Dict[str, Any]] = {}
_job_lock = threading.Lock()
_log_lines: List[str] = []
_LOG_MAX = 500


def _append_log(line: str) -> None:
    _log_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
    if len(_log_lines) > _LOG_MAX:
        del _log_lines[: len(_log_lines) - _LOG_MAX]


def _run_script(job_id: str, cmd: List[str]) -> None:
    with _job_lock:
        _job_store[job_id] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "exit_code": None,
        }
    _append_log(f"[{job_id}] Started: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(_PROJECT_ROOT),
        )
        for line in proc.stdout:
            stripped = line.rstrip()
            if stripped:
                _append_log(f"[{job_id}] {stripped}")
        proc.wait()
        status = "completed" if proc.returncode == 0 else "failed"
        _append_log(f"[{job_id}] Finished with exit code {proc.returncode}")
    except Exception as exc:
        status = "failed"
        _append_log(f"[{job_id}] Error: {exc}")
        proc = None

    with _job_lock:
        _job_store[job_id]["status"] = status
        _job_store[job_id]["finished_at"] = datetime.now().isoformat()
        _job_store[job_id]["exit_code"] = proc.returncode if proc else -1


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class BackfillRequest(BaseModel):
    start_date: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")


class TrainRequest(BaseModel):
    models: List[str] = Field(
        default=["sarima", "lgbm", "tft", "ensemble"],
        description="Models to train",
    )
    skip_mlflow: bool = Field(default=False, description="Skip MLflow logging")


class SystemStatus(BaseModel):
    data_available: bool
    data_rows: int
    data_date_range: Optional[str] = None
    features_available: bool
    features_rows: int
    models_available: List[str]
    results_available: bool
    active_jobs: Dict[str, Any]
    last_training_results: Optional[Dict] = None


class LogsResponse(BaseModel):
    lines: List[str]
    total: int


@router.post("/backfill", response_model=JobResponse, summary="Run data backfill")
def run_backfill(payload: BackfillRequest = BackfillRequest()):
    with _job_lock:
        if "backfill" in _job_store and _job_store["backfill"]["status"] == "running":
            raise HTTPException(400, "Backfill is already running")

    cmd = [sys.executable, "scripts/run_pipeline.py", "--mode", "backfill"]
    if payload.start_date:
        cmd += ["--start", payload.start_date]
    if payload.end_date:
        cmd += ["--end", payload.end_date]

    thread = threading.Thread(target=_run_script, args=("backfill", cmd), daemon=True)
    thread.start()

    return JobResponse(job_id="backfill", status="started", message="Backfill pipeline started")


@router.post("/daily", response_model=JobResponse, summary="Run daily update")
def run_daily():
    with _job_lock:
        if "daily" in _job_store and _job_store["daily"]["status"] == "running":
            raise HTTPException(400, "Daily update is already running")

    cmd = [sys.executable, "scripts/run_pipeline.py", "--mode", "daily"]
    thread = threading.Thread(target=_run_script, args=("daily", cmd), daemon=True)
    thread.start()

    return JobResponse(job_id="daily", status="started", message="Daily update started")


@router.post("/train", response_model=JobResponse, summary="Train models")
def run_training(payload: TrainRequest = TrainRequest()):
    with _job_lock:
        if "train" in _job_store and _job_store["train"]["status"] == "running":
            raise HTTPException(400, "Training is already running")

    allowed = {"sarima", "lgbm", "tft", "ensemble"}
    models = [m for m in payload.models if m in allowed]
    if not models:
        raise HTTPException(400, f"No valid models specified. Choose from: {sorted(allowed)}")

    cmd = [sys.executable, "scripts/train_models.py", "--models"] + models
    if payload.skip_mlflow:
        cmd.append("--no-mlflow")

    thread = threading.Thread(target=_run_script, args=("train", cmd), daemon=True)
    thread.start()

    return JobResponse(job_id="train", status="started", message=f"Training started for: {', '.join(models)}")


@router.get("/status", response_model=SystemStatus, summary="System status")
def get_status(request: Request):
    data_available = PROCESSED_PATH.exists()
    data_rows = 0
    data_range = None
    if data_available:
        try:
            df = pd.read_parquet(PROCESSED_PATH)
            data_rows = len(df)
            dates = pd.to_datetime(df["date"])
            data_range = f"{dates.min().date()} to {dates.max().date()}"
        except Exception:
            pass

    features_available = FEATURES_PATH.exists()
    features_rows = 0
    if features_available:
        try:
            features_rows = len(pd.read_parquet(FEATURES_PATH))
        except Exception:
            pass

    model_files = []
    for p in MODELS_DIR.glob("*.pkl"):
        model_files.append(p.stem)

    results_data = None
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                results_data = json.load(f)
        except Exception:
            pass

    with _job_lock:
        jobs = dict(_job_store)

    return SystemStatus(
        data_available=data_available,
        data_rows=data_rows,
        data_date_range=data_range,
        features_available=features_available,
        features_rows=features_rows,
        models_available=model_files,
        results_available=RESULTS_PATH.exists(),
        active_jobs=jobs,
        last_training_results=results_data,
    )


@router.get("/logs", response_model=LogsResponse, summary="Recent pipeline logs")
def get_logs(tail: int = 100):
    tail = min(tail, _LOG_MAX)
    return LogsResponse(lines=_log_lines[-tail:], total=len(_log_lines))


class MLflowRun(BaseModel):
    run_id: str
    run_name: Optional[str] = None
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    model: Optional[str] = None
    params: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}
    artifact_uri: Optional[str] = None


class MLflowResponse(BaseModel):
    available: bool
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    runs: List[MLflowRun] = []
    total_runs: int = 0


@router.get("/mlflow", response_model=MLflowResponse, summary="MLflow experiment runs")
def get_mlflow_runs(limit: int = 20):
    mlruns_path = _PROJECT_ROOT / "mlruns"
    try:
        import mlflow
        mlflow_available = True
    except ImportError:
        mlflow_available = False

    if not mlflow_available or not mlruns_path.exists():
        return MLflowResponse(available=False)

    tracking_uri = str(mlruns_path)
    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name("thermosense")
        if experiment is None:
            return MLflowResponse(available=True, tracking_uri=tracking_uri)

        runs_data = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )

        runs = []
        for _, row in runs_data.iterrows():
            params = {}
            metrics = {}
            for col in runs_data.columns:
                if col.startswith("params."):
                    key = col[len("params."):]
                    val = row[col]
                    if pd.notna(val):
                        params[key] = val
                elif col.startswith("metrics."):
                    key = col[len("metrics."):]
                    val = row[col]
                    if pd.notna(val):
                        try:
                            metrics[key] = float(val)
                        except (ValueError, TypeError):
                            pass

            start_str = None
            if pd.notna(row.get("start_time")):
                start_str = str(row["start_time"])
            end_str = None
            if pd.notna(row.get("end_time")):
                end_str = str(row["end_time"])

            runs.append(MLflowRun(
                run_id=row["run_id"],
                run_name=row.get("tags.mlflow.runName"),
                status=row.get("status", "UNKNOWN"),
                start_time=start_str,
                end_time=end_str,
                model=params.get("model"),
                params=params,
                metrics=metrics,
                artifact_uri=row.get("artifact_uri"),
            ))

        return MLflowResponse(
            available=True,
            tracking_uri=tracking_uri,
            experiment_name="thermosense",
            runs=runs,
            total_runs=len(runs),
        )
    except Exception:
        return MLflowResponse(available=True, tracking_uri=tracking_uri)
