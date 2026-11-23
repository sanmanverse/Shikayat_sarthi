from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from app.core.config import settings
from app.core.observability import tracer
from app.db.session import init_db
from app.api.endpoints import router as api_router

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()

    if not tracer.is_enabled:
        logger.warning(
            "Langfuse tracing is DISABLED. Set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env to enable observability."
        )

    # Warm up the embedding engine once (builds/loads the cached index; BM25 fallback if unavailable)
    try:
        from app.nlp.embedding_retrieval import get_embedding_engine
        eng = get_embedding_engine()
        logger.info("Embedding retrieval: %s", "ready" if (eng and eng.is_ready) else "unavailable (BM25 only)")
    except Exception as exc:
        logger.warning("Embedding warm-up failed: %s", exc)

    # APScheduler for SLA scans
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from app.db.session import SessionLocal
        from app.services.sla_predictor import sla_predictor
        from app.db.models import Grievance

        scheduler = AsyncIOScheduler()

        def _sla_scan_job():
            db = SessionLocal()
            try:
                open_cases = db.query(Grievance).filter(
                    Grievance.status.in_(["RECEIVED", "TRIAGED", "IN_PROGRESS", "DISPUTED_REOPENED"])
                ).all()
                for g in open_cases:
                    sla_predictor.evaluate_case_sla(g, db)
                logger.debug("SLA scan: checked %d open cases.", len(open_cases))
            finally:
                db.close()

        scheduler.add_job(
            _sla_scan_job,
            IntervalTrigger(minutes=settings.SLA_SCAN_INTERVAL_MINUTES),
            id="sla_scan",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("APScheduler started (SLA scan every %d min).", settings.SLA_SCAN_INTERVAL_MINUTES)
    except Exception as exc:
        logger.warning("APScheduler failed to start: %s", exc)
        scheduler = None

    yield

    # Shutdown
    if scheduler:
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
    tracer.shutdown()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.APP_VERSION,
    description="Intelligent Grievance Redressal Multi-Agent System (HackFusion 4.0)",
    lifespan=lifespan,
)

# CORS: restricted origins — never combine allow_origins=["*"] with allow_credentials=True
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "docs_url": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
