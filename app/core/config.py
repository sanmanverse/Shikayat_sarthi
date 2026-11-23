"""
Central configuration via pydantic-settings.
All secrets come from environment variables / .env file — NEVER hard-coded.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "HackFusion Grievance Redressal System"
    APP_VERSION: str = "2.1.0"
    DATABASE_URL: str = "sqlite:///./grievance_system.db"

    # --- Secrets (loaded from .env) ---
    SARVAM_API_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_PROJECT_ID: str = ""
    LANGFUSE_SALT: str = "hackfusion-pii-salt-change-in-prod"

    # Auth (demo keys — change in production)
    DEMO_OFFICER_KEY: str = "officer-demo-key"
    DEMO_CITIZEN_KEY: str = "citizen-demo-key"
    JWT_SECRET: str = "hackfusion-jwt-secret-change-in-prod"

    # --- Sarvam (verified against docs.sarvam.ai) ---
    # Chat: POST https://api.sarvam.ai/v1/chat/completions serves sarvam-105b and
    # sarvam-105b-conversations. Auth header: api-subscription-key.
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_CHAT_MODEL: str = "sarvam-105b"
    # STT: POST /speech-to-text (multipart). saaras:v3 supports `mode` (transcribe/translate/...).
    SARVAM_STT_MODEL: str = "saaras:v3"
    SARVAM_STT_MODE: str = "transcribe"
    SARVAM_STT_LANGUAGE: str = ""            # "" = let Sarvam auto-detect; else e.g. "mr-IN"
    SARVAM_MAX_TOKENS: int = 1500            # reasoning tokens are billed as completion tokens
    SARVAM_TIMEOUT_S: int = 30
    SARVAM_MAX_RETRIES: int = 1
    SARVAM_REASONING_EFFORT: str = ""        # "" = do not send; else low|medium|high

    # --- Guarded agent loop ---
    AGENT_LOOP_ENABLED: bool = True          # only active when a Sarvam key is configured
    AGENT_MAX_STEPS: int = 5

    # --- Policy Thresholds ---
    AUTO_HANDLE_CONFIDENCE_THRESHOLD: float = 0.75
    MIN_RETRIEVAL_SIMILARITY: float = 0.60

    # --- Clustering / Duplicate thresholds ---
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.82
    CLUSTER_WINDOW_DAYS: int = 15
    CLUSTER_LOCALITY_FUZZY_THRESHOLD: int = 80  # token-set ratio threshold (0-100)

    # --- SLA Scheduler ---
    SLA_SCAN_INTERVAL_MINUTES: int = 5

    # --- Dev / Demo ---
    ENABLE_DEV_BACKDATE: bool = False

    # --- CORS (restrict origins; never combine * with credentials) ---
    CORS_ORIGINS: List[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # --- Embedding / retrieval ---
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    EMBEDDING_ALLOW_DOWNLOAD: bool = True    # False = never touch the network (tests, offline demos)
    # Cosine thresholds for e5-small. UNCALIBRATED DEFAULTS: run scripts/calibrate_retrieval.py
    # on a machine with the model and put the recommended values in .env.
    EMBED_STRONG_SIM: float = 0.84
    EMBED_WEAK_SIM: float = 0.78
    RRF_K: int = 60
    # A sensitive precedent that is nearly as similar as the best match counts as a safety signal.
    SENSITIVE_PROXIMITY_RATIO: float = 0.90   # BM25-only mode: weak backstop; the text guard + stage B are the real nets
    SENSITIVE_PROXIMITY_GAP: float = 0.03     # used with embeddings: cosine gap to the best match (uncalibrated)

    # --- Predictive SLA (empirical, from mock_grievance_history durations) ---
    SLA_ALERT_THRESHOLD: float = 0.75
    SLA_MIN_SAMPLES: int = 5

    # --- Tracing visibility ---
    TRACE_PUBLIC_NON_SENSITIVE: bool = True   # mark non-sensitive traces public (shareable link)
    TRACE_PUBLIC_FOR_SENSITIVE: bool = False  # sensitive traces stay private unless explicitly enabled

    # --- Data paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


settings = Settings()
