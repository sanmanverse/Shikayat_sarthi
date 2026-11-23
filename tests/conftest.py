import os
# Tests must never touch the network (no model download, no Sarvam, no Langfuse).
os.environ.setdefault("EMBEDDING_ALLOW_DOWNLOAD", "false")
os.environ.setdefault("SARVAM_API_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
import pytest
from app.db.session import init_db, SessionLocal

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()

@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
