import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.config import IS_POSTGRES, ALLOWED_ORIGINS, DEMO_MODE
from backend.app.storage.database import init_db, get_db, fail_stale_training_jobs

from backend.app.api.v1.auth import router as auth_router
from backend.app.api.v1.stations import router as stations_router
from backend.app.api.v1.telemetry import router as telemetry_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.drift import router as drift_router
from backend.app.api.v1.evidence import router as evidence_router
from backend.app.api.v1.faults import router as faults_router
from backend.app.api.v1.incidents import router as incidents_router
from backend.app.api.v1.maintenance import router as maintenance_router
from backend.app.services.weather_service import weather_service
from backend.app.services import drift_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("skyguard.backend")

async def _drift_nightly_loop():
    """Nightly drift job (spec §21.5, §28.5): compute drift for every
    station, store it, and — outside DEMO_MODE — queue retraining
    candidates / open DATA_QUALITY incidents on MODERATE/HIGH. In
    DEMO_MODE the job only observes (§22.2: automated triggers are
    disabled in demo mode); the admin's on-demand POST /drift/compute
    still performs the actions, because a human explicitly asked.
    The job never blocks the poller: one station's failure is logged
    and the loop continues, and the compute runs in a worker thread.
    """
    await asyncio.sleep(300)  # let the server finish starting up
    while True:
        try:
            outcomes = await asyncio.to_thread(
                drift_service.compute_all_stations,
                perform_actions=not DEMO_MODE)
            logger.info("Nightly drift computation finished: %s", outcomes)
        except Exception as error:  # noqa: BLE001 - the loop must survive
            logger.exception("Nightly drift computation failed: %s", error)
        await asyncio.sleep(24 * 3600)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Database & Schema
    db_type = "Cloud PostgreSQL / TimescaleDB" if IS_POSTGRES else "SQLite (WAL Mode)"
    logger.info(f"Initializing SkyGuard-AI Database ({db_type})...")
    init_db()
    logger.info(f"Database schema initialized successfully ({db_type}).")
    stale = fail_stale_training_jobs()
    if stale:
        logger.warning(f"Marked {stale} training job(s) left RUNNING by a previous process as FAILED (interrupted by restart).")
    
    # Start Background Weather Poller
    weather_task = asyncio.create_task(weather_service.poll_loop()) if os.getenv("WEATHER_POLL_ENABLED", "true").lower() == "true" else None

    # Nightly model-drift job (spec §21.5): first run 5 minutes after
    # startup, then once a day. See _drift_nightly_loop for the
    # DEMO_MODE rule (observe only in demo mode).
    drift_task = asyncio.create_task(_drift_nightly_loop())

    yield
    # Shutdown
    logger.info("SkyGuard-AI Backend service shutting down.")
    weather_service.stop()
    for task in (weather_task, drift_task):
        if not task:
            continue
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            pass

app = FastAPI(
    title="SkyGuard-AI Tactical Backend Service",
    description="Cloud PostgreSQL & TimescaleDB Telemetry Ingestion, Station Provisioning & Station-Adaptive MLOps Pipeline",
    version="2.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
# IMPORTANT: telemetry_router must be registered before stations_router.
# stations_router has a wildcard route `GET /stations/{station_id}` that would
# shadow `/stations/{station_id}/telemetry/stats` if mounted first.
app.include_router(auth_router, prefix="/api/v1")
app.include_router(telemetry_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(drift_router, prefix="/api/v1")
app.include_router(evidence_router, prefix="/api/v1")
app.include_router(faults_router, prefix="/api/v1")
app.include_router(incidents_router, prefix="/api/v1")
app.include_router(maintenance_router, prefix="/api/v1")
app.include_router(stations_router, prefix="/api/v1")

@app.get("/api/v1/health", tags=["Diagnostics"])
def health_check():
    try:
        with get_db() as conn:
            conn.cursor().execute("SELECT 1")
    except Exception as error:
        logger.warning("Database health check failed: %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Database unavailable") from error
    return {
        "status": "ONLINE",
        "service": "SkyGuard-AI Backend",
        "version": "2.1.0",
        "database": "Cloud PostgreSQL / TimescaleDB" if IS_POSTGRES else "SQLite (WAL Mode)",
        "demo_mode": DEMO_MODE,
        "security": "Argon2id password hashes / JWT HS256",
        "mlops": "Station-Adaptive Isolation Forest"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=False)
