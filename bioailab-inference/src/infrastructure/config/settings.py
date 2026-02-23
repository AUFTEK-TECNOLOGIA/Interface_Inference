"""
Configurações da aplicação.

Responsabilidade única: centralizar configurações do ambiente.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Settings:
    """Configurações da aplicação."""
    
    # Paths
    base_dir: Path
    resources_dir: Path
    
    # Segurança
    api_token: str
    
    # MongoDB
    mongo_uri: str
    tenant_db_prefix: str
    
    # API de conversão espectral
    spectral_api_url: str
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Cria configurações a partir de variáveis de ambiente."""
        base_dir = Path(__file__).resolve().parents[3]
        
        return cls(
            base_dir=base_dir,
            resources_dir=base_dir / "resources",
            api_token=os.getenv("TOKEN", "1337"),
            mongo_uri=os.getenv(
                "MONGO_URI",
                "mongodb+srv://golang:s0meyLEQavWmUBmx@iot.2ypazq2.mongodb.net/?retryWrites=true&w=majority&appName=iot"
            ),
            tenant_db_prefix=os.getenv("TENANT_DB_PREFIX", "bioailab_"),
            spectral_api_url=os.getenv(
                "SPECTRAL_API_URL",
                "https://spectral.bioailab.com.br/convert"
            ),
        )


@lru_cache()
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings.from_env()
