"""
Lightweight Authentication & Role-Based Access Control.
Group 4:
- Supports API keys (X-API-Key header) or Bearer tokens.
- Roles: CITIZEN, OFFICER, SUPERVISOR, SYSTEM.
- Default demo keys configurable via settings.
"""
from typing import Optional
from dataclasses import dataclass
from fastapi import Header, HTTPException, Depends
from app.core.config import settings

@dataclass
class AuthUser:
    username: str
    role: str  # "CITIZEN", "OFFICER", "SUPERVISOR", "SYSTEM"
    is_authenticated: bool = True

def get_current_user(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> AuthUser:
    """
    Extracts authenticated user from API Key header or Authorization header.
    Defaults to demo roles if headers match settings, or citizen if absent for public endpoints.
    """
    token = x_api_key
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()

    if token:
        if token == settings.DEMO_OFFICER_KEY or token == "officer-key":
            return AuthUser(username="nodal_officer_demo", role="OFFICER")
        if token == "supervisor-key" or token == "admin-key":
            return AuthUser(username="district_supervisor_demo", role="SUPERVISOR")
        if token == "system-key":
            return AuthUser(username="ai_system_agent", role="SYSTEM")
        if token == settings.DEMO_CITIZEN_KEY or token == "citizen-key":
            return AuthUser(username="citizen_demo", role="CITIZEN")

    # If no key provided, default to CITIZEN role (for citizen-facing endpoints)
    return AuthUser(username="anonymous_citizen", role="CITIZEN", is_authenticated=False)


def require_officer(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    """Enforces that caller has OFFICER or SUPERVISOR role."""
    if user.role not in ("OFFICER", "SUPERVISOR"):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: This action requires an OFFICER or SUPERVISOR role. Provide X-API-Key."
        )
    return user
