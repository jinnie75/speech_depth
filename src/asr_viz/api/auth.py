from __future__ import annotations

from dataclasses import dataclass
import json
from urllib.error import URLError
from urllib.request import urlopen
import uuid

from fastapi import Header, HTTPException, Request, Response, status

from asr_viz.core.settings import settings


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    auth_provider: str


_jwks_client = None


async def attach_current_user(request: Request, call_next):
    current_user, cookie_value = _resolve_request_user(request)
    request.state.authenticated_user = current_user

    response = await call_next(request)
    if cookie_value is not None:
        _set_anonymous_session_cookie(response, cookie_value)
    return response


def require_current_user(request: Request) -> AuthenticatedUser:
    current_user = getattr(request.state, "authenticated_user", None)
    if not isinstance(current_user, AuthenticatedUser):
        raise HTTPException(status_code=500, detail="authenticated user was not attached to the request")
    return current_user


def _resolve_request_user(request: Request) -> tuple[AuthenticatedUser, str | None]:
    authorization = request.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        return _authenticate_bearer_token(authorization), None

    if settings.require_auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")

    dev_header_user = request.headers.get("x-asr-viz-user-id")
    if isinstance(dev_header_user, str) and dev_header_user.strip():
        return AuthenticatedUser(user_id=dev_header_user.strip(), auth_provider="dev-header"), None

    anonymous_cookie_name = settings.anonymous_session_cookie_name
    existing_cookie = request.cookies.get(anonymous_cookie_name)
    if isinstance(existing_cookie, str) and existing_cookie.strip():
        return AuthenticatedUser(user_id=existing_cookie.strip(), auth_provider="anonymous-cookie"), None

    cookie_value = f"anon_{uuid.uuid4()}"
    return AuthenticatedUser(user_id=cookie_value, auth_provider="anonymous-cookie"), cookie_value


def _set_anonymous_session_cookie(response: Response, cookie_value: str) -> None:
    secure = settings.anonymous_session_cookie_secure or settings.app_env == "production"
    same_site = "none" if secure else "lax"
    response.set_cookie(
        key=settings.anonymous_session_cookie_name,
        value=cookie_value,
        httponly=True,
        max_age=settings.anonymous_session_cookie_max_age_seconds,
        secure=secure,
        samesite=same_site,
        domain=settings.anonymous_session_cookie_domain,
        path="/",
    )


def _authenticate_bearer_token(authorization: str) -> AuthenticatedUser:
    if settings.auth_provider != "clerk":
        raise HTTPException(status_code=500, detail="bearer auth is not configured")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")

    claims = _verify_clerk_token(token)
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token missing subject")

    return AuthenticatedUser(user_id=subject.strip(), auth_provider="clerk")


def _verify_clerk_token(token: str) -> dict:
    try:
        import jwt
        from jwt import PyJWKClient
        from jwt.exceptions import InvalidTokenError
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="PyJWT is required for Clerk authentication. Install project dependencies before enabling Clerk auth.",
        ) from exc

    issuer = settings.clerk_issuer_url
    if not issuer:
        raise HTTPException(status_code=500, detail="CLERK_ISSUER_URL must be configured for Clerk auth")

    jwks_url = settings.clerk_jwks_url or f"{issuer.rstrip('/')}/.well-known/jwks.json"
    audience = settings.clerk_audience

    try:
        jwks_client = _get_jwks_client(PyJWKClient, jwks_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        verification_options = {"verify_aud": bool(audience)}
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options=verification_options,
        )
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid bearer token") from exc


def _get_jwks_client(pyjwk_client_type, jwks_url: str):
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = pyjwk_client_type(jwks_url)
    return _jwks_client


def fetch_clerk_jwks_document() -> dict:
    issuer = settings.clerk_issuer_url
    if not issuer:
        raise RuntimeError("CLERK_ISSUER_URL is not configured")

    jwks_url = settings.clerk_jwks_url or f"{issuer.rstrip('/')}/.well-known/jwks.json"
    try:
        with urlopen(jwks_url, timeout=settings.clerk_network_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to fetch Clerk JWKS from {jwks_url}") from exc
