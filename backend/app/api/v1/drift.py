"""Model drift endpoints (spec §21.5, §25.3).

  GET  /stations/{id}/drift?days=90          latest + history (sparklines)
  GET  /stations/{id}/drift/latest           latest per model type
  POST /stations/{id}/drift/compute          admin-only on-demand compute
  POST /stations/{id}/drift/queue-retraining admin-only manual queue

Read access follows the station rule (admin any station, operator
only their own); on-demand compute and manual queueing are admin-only
— computing drift is CPU work, and queueing a candidate is a
governance decision. Computing drift never trains, activates or
deactivates anything.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional

from backend.app.api.v1.auth import get_current_user, require_admin, require_station_access
from backend.app.services import drift_service
from backend.app.storage.database import get_db
from ml.drift_engine import threshold_table

router = APIRouter(tags=["Model Drift Monitoring"])


def _require_station_exists(station_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT station_id FROM stations WHERE station_id = ?",
                    (station_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Station not found")


@router.get("/stations/{station_id}/drift")
def get_station_drift(
    station_id: str,
    days: int = 90,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Latest drift per model type plus history for the sparklines."""
    clean_id = station_id.strip().upper()
    require_station_access(clean_id, current_user)
    _require_station_exists(clean_id)
    latest = drift_service.latest_drift(clean_id)
    history = drift_service.drift_history(clean_id, days=max(1, min(days, 365)))
    candidates = drift_service.list_retraining_candidates(clean_id)
    return {
        "success": True,
        "station_id": clean_id,
        "latest": latest,
        "history": history,
        "retraining_candidates": candidates,
        "thresholds": threshold_table(),
        "window_days": drift_service.WINDOW_DAYS,
    }


@router.get("/stations/{station_id}/drift/latest")
def get_latest_drift(
    station_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Latest computed drift per model type (no history)."""
    clean_id = station_id.strip().upper()
    require_station_access(clean_id, current_user)
    _require_station_exists(clean_id)
    return {
        "success": True,
        "station_id": clean_id,
        "latest": drift_service.latest_drift(clean_id),
        "thresholds": threshold_table(),
        "window_days": drift_service.WINDOW_DAYS,
    }


@router.post("/stations/{station_id}/drift/compute")
def compute_drift(
    station_id: str,
    admin_user: Dict[str, Any] = Depends(require_admin),
):
    """Compute drift now for one station (admin).

    This is the manual trigger: it computes, stores, and — when the
    severity is MODERATE or HIGH — queues a retraining candidate and,
    on HIGH, opens a DATA_QUALITY incident. It never trains or
    activates a model.
    """
    clean_id = station_id.strip().upper()
    try:
        result = drift_service.compute_station_drift(clean_id, perform_actions=True)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"success": True, **result}


@router.post("/stations/{station_id}/drift/queue-retraining")
def queue_drift_retraining(
    station_id: str,
    payload: Optional[Dict[str, Any]] = None,
    admin_user: Dict[str, Any] = Depends(require_admin),
):
    """Manually queue a retraining candidate for a drifted model.

    The candidate is a queue entry only: training it still needs a
    human to start the job, and activation still needs the promotion
    gate. Never trains inline — nightly jobs should not be triggered
    from a click during a demo (§21.5).
    """
    clean_id = station_id.strip().upper()
    body = payload or {}
    model_version = str(body.get("model_version") or "").strip()
    if not model_version:
        raise HTTPException(status_code=400, detail="model_version is required")
    _require_station_exists(clean_id)
    priority = str(body.get("priority") or "LOW").strip().upper()
    if priority not in ("LOW", "HIGH"):
        priority = "LOW"
    candidate, created = drift_service.queue_retraining_candidate(
        clean_id, model_version,
        str(body.get("trigger_reason") or "MANUAL_DRIFT_REVIEW"),
        priority, actor=admin_user.get("sub") or admin_user.get("username"))
    return {"success": True, "station_id": clean_id,
            "candidate": candidate, "created": created}
