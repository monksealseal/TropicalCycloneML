"""
OncologyML API — FastAPI web service for the cancer research agent.

Provides async job submission, status polling, and result retrieval
for autonomous cancer dataset ML analysis.
"""

import uuid
import threading
import traceback
from datetime import datetime, timezone
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cancer_agent.agent import CancerResearchAgent, DISCLAIMER, STAGES
from cancer_agent.data.loader import DATASET_REGISTRY
from cancer_agent.evaluation.reporter import Reporter
from cancer_agent.models.trainer import MODEL_CONFIGS


# ---------------------------------------------------------------------------
# Job store (in-memory — swap for Redis / Postgres in production)
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    dataset: str = Field(
        default="breast_cancer",
        description="Dataset name to analyse",
    )
    models: list[str] | None = Field(
        default=None,
        description="List of model names (null = all)",
    )
    scaling: str = Field(default="standard", pattern="^(standard|minmax)$")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=20)


class JobOut(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    message: str


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    completed_at: str | None = None
    stage: str | None = None
    stage_number: int | None = None
    total_stages: int | None = None
    progress_pct: int | None = None
    report: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Background runner
# ---------------------------------------------------------------------------
def _make_progress_callback(job_id: str):
    """Return a callback that updates the job's stage info."""
    def on_progress(stage_index: int, total: int, stage_name: str):
        _jobs[job_id]["stage"] = stage_name
        _jobs[job_id]["stage_number"] = stage_index
        _jobs[job_id]["total_stages"] = total
        _jobs[job_id]["progress_pct"] = int((stage_index / total) * 100)
    return on_progress


def _run_job(job_id: str, request: AnalysisRequest):
    """Execute analysis in a background thread."""
    _jobs[job_id]["status"] = JobStatus.running
    try:
        agent = CancerResearchAgent(
            dataset=request.dataset,
            models=request.models,
            scaling=request.scaling,
            test_size=request.test_size,
            cv_folds=request.cv_folds,
            output_dir=f"/tmp/oncologyml_jobs/{job_id}",
            on_progress=_make_progress_callback(job_id),
        )
        report = agent.run()
        # Ensure all numpy types are converted to native Python for JSON
        reporter = Reporter()
        report = reporter._make_serializable(report)
        _jobs[job_id]["status"] = JobStatus.completed
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["report"] = report
        _jobs[job_id]["progress_pct"] = 100
    except Exception:
        _jobs[job_id]["status"] = JobStatus.failed
        _jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        _jobs[job_id]["error"] = traceback.format_exc()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="OncologyML",
        description=(
            "Autonomous cancer research ML analysis as a service. "
            "Submit datasets, get comprehensive ML reports. "
            f"\n\n**{DISCLAIMER}**"
        ),
        version="0.1.0",
    )

    import pathlib
    static_dir = pathlib.Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # -----------------------------------------------------------------------
    # Landing page
    # -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def landing():
        html_file = static_dir / "index.html"
        if html_file.exists():
            return HTMLResponse(html_file.read_text())
        return HTMLResponse("<h1>OncologyML API</h1><p>Visit /docs for API docs.</p>")

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "OncologyML", "version": "0.1.0"}

    # -----------------------------------------------------------------------
    # List available resources
    # -----------------------------------------------------------------------
    @app.get("/api/v1/datasets")
    async def list_datasets():
        return {
            name: {"description": info["description"], "task": info["task"]}
            for name, info in DATASET_REGISTRY.items()
        }

    @app.get("/api/v1/models")
    async def list_models():
        return {
            name: {"class": cls.__name__}
            for name, (cls, _) in MODEL_CONFIGS.items()
        }

    # -----------------------------------------------------------------------
    # Submit analysis job
    # -----------------------------------------------------------------------
    @app.post("/api/v1/analyze", response_model=JobOut, status_code=202)
    async def submit_analysis(request: AnalysisRequest):
        if request.dataset not in DATASET_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown dataset: {request.dataset}. "
                       f"Available: {list(DATASET_REGISTRY.keys())}",
            )
        if request.models:
            unknown = set(request.models) - set(MODEL_CONFIGS.keys())
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown models: {unknown}. "
                           f"Available: {list(MODEL_CONFIGS.keys())}",
                )

        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        _jobs[job_id] = {
            "status": JobStatus.pending,
            "created_at": now,
            "completed_at": None,
            "stage": None,
            "stage_number": 0,
            "total_stages": len(STAGES),
            "progress_pct": 0,
            "report": None,
            "error": None,
            "request": request.model_dump(),
        }

        thread = threading.Thread(target=_run_job, args=(job_id, request), daemon=True)
        thread.start()

        return JobOut(
            job_id=job_id,
            status=JobStatus.pending,
            created_at=now,
            message="Analysis job submitted. Poll /api/v1/jobs/{job_id} for status.",
        )

    # -----------------------------------------------------------------------
    # Job status & results
    # -----------------------------------------------------------------------
    @app.get("/api/v1/jobs/{job_id}", response_model=JobResult)
    async def get_job(job_id: str):
        if job_id not in _jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = _jobs[job_id]
        return JobResult(
            job_id=job_id,
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job["completed_at"],
            stage=job.get("stage"),
            stage_number=job.get("stage_number"),
            total_stages=job.get("total_stages"),
            progress_pct=job.get("progress_pct"),
            report=job["report"] if job["status"] == JobStatus.completed else None,
            error=job["error"] if job["status"] == JobStatus.failed else None,
        )

    @app.get("/api/v1/jobs")
    async def list_jobs(
        status: JobStatus | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=100),
    ):
        items = []
        for jid, job in sorted(
            _jobs.items(), key=lambda x: x[1]["created_at"], reverse=True
        ):
            if status and job["status"] != status:
                continue
            items.append({
                "job_id": jid,
                "status": job["status"],
                "created_at": job["created_at"],
                "completed_at": job["completed_at"],
                "stage": job.get("stage"),
                "progress_pct": job.get("progress_pct"),
                "dataset": job["request"]["dataset"],
            })
            if len(items) >= limit:
                break
        return {"jobs": items, "total": len(_jobs)}

    return app
