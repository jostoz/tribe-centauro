"""Store SQLite de anuncios y entidades.

Reemplaza los ``records/*.json`` sueltos: consultable, incremental y con las
entidades canónicas persistidas (para deduplicar el grafo y poder agregar por
entidad sin reprocesar).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

DB_PATH = Path("data/discovery/centauro.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ads (
    id            TEXT PRIMARY KEY,
    source        TEXT,
    url           TEXT,
    title         TEXT,
    duration      REAL,
    channel       TEXT,
    upload_date   TEXT,
    view_count    INTEGER,
    like_count    INTEGER,
    comment_count INTEGER,
    stats_updated_at TEXT,
    video_path    TEXT,
    transcript    TEXT,
    understanding TEXT,
    neural        TEXT,
    updated_at    TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS entities (
    ad_id     TEXT,
    kind      TEXT,
    canonical TEXT,
    raw       TEXT,
    PRIMARY KEY (ad_id, kind, canonical)
);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(kind, canonical);
"""

_COLS = (
    "id", "source", "url", "title", "duration", "channel",
    "upload_date", "view_count", "video_path",
)

# Columnas añadidas después de la primera versión: se migran con ALTER TABLE
# (CREATE TABLE IF NOT EXISTS no las añadiría a una tabla ya existente).
_MIGRATIONS = {
    "like_count": "INTEGER",
    "comment_count": "INTEGER",
    "stats_updated_at": "TEXT",
}


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    return con


def init(con: sqlite3.Connection) -> None:
    con.executescript(_SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(ads)")}
    for col, typ in _MIGRATIONS.items():
        if col not in cols:
            con.execute(f"ALTER TABLE ads ADD COLUMN {col} {typ}")
    con.commit()


def upsert_ad(con: sqlite3.Connection, rec: dict, entities: Optional[Iterable[Tuple[str, str, str]]] = None) -> None:
    """Inserta/actualiza un anuncio sin pisar campos ya calculados con None."""
    row = {c: rec.get(c) for c in _COLS}
    for jf in ("transcript", "understanding", "neural"):
        v = rec.get(jf)
        row[jf] = json.dumps(v, ensure_ascii=False) if v is not None else None
    con.execute(
        f"""
        INSERT INTO ads ({", ".join(row)})
        VALUES ({", ".join("?" for _ in row)})
        ON CONFLICT(id) DO UPDATE SET
            url=COALESCE(excluded.url, ads.url),
            title=COALESCE(excluded.title, ads.title),
            duration=COALESCE(excluded.duration, ads.duration),
            channel=COALESCE(excluded.channel, ads.channel),
            upload_date=COALESCE(excluded.upload_date, ads.upload_date),
            view_count=COALESCE(excluded.view_count, ads.view_count),
            video_path=COALESCE(excluded.video_path, ads.video_path),
            transcript=COALESCE(excluded.transcript, ads.transcript),
            understanding=COALESCE(excluded.understanding, ads.understanding),
            neural=COALESCE(excluded.neural, ads.neural),
            updated_at=datetime('now')
        """,
        list(row.values()),
    )
    if entities is not None:
        con.execute("DELETE FROM entities WHERE ad_id=?", (rec["id"],))
        con.executemany(
            "INSERT OR REPLACE INTO entities (ad_id, kind, canonical, raw) VALUES (?,?,?,?)",
            [(rec["id"], k, c, r) for k, c, r in entities],
        )
    con.commit()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for jf in ("transcript", "understanding", "neural"):
        if d.get(jf):
            d[jf] = json.loads(d[jf])
    return d


def upsert_stats(con: sqlite3.Connection, rec: dict) -> None:
    """Actualiza las métricas públicas (vistas/likes/comments) sin tocar el resto.

    Se guardan **crudas** (con ``stats_updated_at``); las derivadas (vistas/día, tasas)
    se calculan al leer con :func:`stats_table`, porque dependen de la fecha de consulta.
    """
    con.execute(
        """
        UPDATE ads SET
            view_count=COALESCE(?, view_count),
            like_count=COALESCE(?, like_count),
            comment_count=COALESCE(?, comment_count),
            upload_date=COALESCE(?, upload_date),
            duration=COALESCE(?, duration),
            channel=COALESCE(?, channel),
            title=COALESCE(?, title),
            stats_updated_at=datetime('now')
        WHERE id=?
        """,
        (
            rec.get("view_count"), rec.get("like_count"), rec.get("comment_count"),
            rec.get("upload_date"), rec.get("duration"), rec.get("channel"),
            rec.get("title"), rec["id"],
        ),
    )
    con.commit()


def stats_table(con: sqlite3.Connection, now=None) -> List[dict]:
    """Métricas públicas con derivadas: vistas/día, tasa de likes y de comentarios.

    ``now`` es inyectable para poder testear el cálculo sin depender del reloj.
    """
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    out: List[dict] = []
    for a in all_ads(con):
        views, likes, comments = a.get("view_count"), a.get("like_count"), a.get("comment_count")
        days = None
        ud = a.get("upload_date")
        if ud:
            try:
                up = datetime.strptime(ud, "%Y%m%d").replace(tzinfo=timezone.utc)
                days = max((now - up).days, 1)
            except ValueError:
                days = None
        out.append(
            {
                "id": a["id"],
                "channel": a.get("channel"),
                "views": views,
                "likes": likes,
                "comments": comments,
                "days": days,
                "views_per_day": (views / days) if (views and days) else None,
                "like_rate": (likes / views) if (likes and views) else None,
                "comment_rate": (comments / views) if (comments and views) else None,
                "stats_updated_at": a.get("stats_updated_at"),
            }
        )
    return out


def get_ad(con: sqlite3.Connection, ad_id: str) -> Optional[dict]:
    cur = con.execute("SELECT * FROM ads WHERE id=?", (ad_id,))
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def all_ads(con: sqlite3.Connection) -> List[dict]:
    return [_row_to_dict(r) for r in con.execute("SELECT * FROM ads ORDER BY id")]


def top_entities(con: sqlite3.Connection, kind: Optional[str] = None, limit: int = 20) -> List[Tuple[str, int]]:
    """Entidades más frecuentes (nº de anuncios distintos), para inspección rápida."""
    q = "SELECT canonical, COUNT(DISTINCT ad_id) n FROM entities"
    args: list = []
    if kind:
        q += " WHERE kind=?"
        args.append(kind)
    q += " GROUP BY canonical ORDER BY n DESC, canonical LIMIT ?"
    args.append(limit)
    return [(r[0], r[1]) for r in con.execute(q, args)]
