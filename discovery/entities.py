"""Normalización/canonicalización de entidades para deduplicar el grafo.

El VLM devuelve variantes de la misma entidad ("logo Telcel", "logotipo de Telcel",
"Logotipo"). Sin canonicalizar, el grafo se fragmenta en nodos casi-duplicados y las
aristas anuncio↔anuncio no aparecen. Aquí se unifican a una forma canónica.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

# clave en el JSON del VLM -> prefijo de nodo en el grafo
ATTR_KINDS = [
    ("objetos", "obj"),
    ("personajes", "pers"),
    ("temas", "tema"),
    ("marca_elementos", "marca"),
]

_STOP = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "con", "y", "o", "u", "para", "por", "al", "a",
}

# sinónimos frecuentes en descripciones publicitarias -> forma canónica
_ALIASES: Dict[str, str] = {
    "logotipo": "logo",
    "logotipos": "logo",
    "marca": "logo",
    "logotipo telcel": "logo telcel",
    "logo de telcel": "logo telcel",
    "logotipo de telcel": "logo telcel",
    "telefono celular": "telefono",
    "telefono movil": "telefono",
    "celular": "telefono",
    "movil": "telefono",
    "smartphone": "telefono",
    "automovil": "coche",
    "auto": "coche",
    "carro": "coche",
    "vehiculo": "coche",
    "personas": "persona",
    "gente": "persona",
    "hombres": "hombre",
    "mujeres": "mujer",
    "ninos": "nino",
    "colores": "color",
}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def canonical(value: str) -> str:
    """Forma canónica de una entidad (minúsculas, sin acentos/puntuación, sin artículos)."""
    v = _strip_accents(str(value).lower())
    v = re.sub(r"[^\w\s]", " ", v)
    words = [w for w in v.split() if w and w not in _STOP]
    v = " ".join(words)
    return _ALIASES.get(v, v)


def extract_entities(understanding: dict) -> List[Tuple[str, str, str]]:
    """Devuelve ``[(kind, canonical, raw), ...]`` desde el JSON del VLM.

    ``raw`` es la primera forma legible vista, para mostrarla como etiqueta.
    """
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for key, kind in ATTR_KINDS:
        for item in understanding.get(key) or []:
            if not isinstance(item, str) or not item.strip():
                continue
            canon = canonical(item)
            if not canon:
                continue
            sig = (kind, canon)
            if sig in seen:
                continue
            seen.add(sig)
            out.append((kind, canon, item.strip()))
    return out
