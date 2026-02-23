"""
Dependências da API.
"""

from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_400_BAD_REQUEST

from ..infrastructure.config.settings import get_settings


api_key_header = APIKeyHeader(name="token", auto_error=False)


def validate_api_key(header: Optional[str] = Security(api_key_header)) -> bool:
    """Valida o token de API."""
    settings = get_settings()
    
    if header is None:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="API key is missing"
        )
    
    if header != settings.api_token:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Access not allowed"
        )
    
    return True
