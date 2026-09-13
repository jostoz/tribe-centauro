"""
Aplicación FastAPI — adaptador REST sobre ``service/``.

El modelo NO se carga en el arranque: si el checkpoint o el acceso a HuggingFace no
están disponibles, el proceso debe seguir sirviendo ``/health`` y devolviendo errores
accionables en lugar de morir al importar.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import assert_license_allows_use, license_mode, router

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"


def _allowed_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # El gate de licencia se evalúa al arrancar: mejor fallar pronto y claro.
    assert_license_allows_use()
    logger.info("TRIBE Ads API v%s — LICENSE_MODE=%s", VERSION, license_mode())
    yield
    logger.info("Cerrando TRIBE Ads API")


app = FastAPI(
    title="TRIBE Ads API",
    description=(
        "Análisis de activación neural de creativos publicitarios con TRIBE v2. "
        "Los valores están en unidades crudas del modelo y NO están calibrados "
        "contra resultados de campaña."
    ),
    version=VERSION,
    lifespan=lifespan,
)

# Nunca "*": el endpoint no debe ser invocable desde cualquier origen.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "TRIBE Ads API",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
        "license_mode": license_mode(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEBUG", "False").lower() == "true",
    )
