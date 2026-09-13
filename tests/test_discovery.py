"""Tests de la lógica de escalabilidad: caché, entidades, store y grafo.

Deterministas y sin GPU (no cargan modelos). Defienden los contratos de los que
depende la idempotencia y el grafo de contenidos.
"""

from __future__ import annotations

from discovery import cache, entities, graph, store


# --- caché content-addressed ------------------------------------------------

def test_content_key_is_stable_and_sensitive(tmp_path, monkeypatch):
    monkeypatch.setenv("CENTAURO_CACHE", str(tmp_path / "cache"))
    video = tmp_path / "a.mp4"
    video.write_bytes(b"contenido de prueba")

    k1 = cache.content_key(video, "transcribe", "whisper|es")
    k2 = cache.content_key(video, "transcribe", "whisper|es")
    assert k1 == k2, "misma entrada -> misma clave (idempotencia)"

    assert cache.content_key(video, "understand", "whisper|es") != k1, "cambia la etapa"
    assert cache.content_key(video, "transcribe", "otro|es") != k1, "cambian los params"

    other = tmp_path / "b.mp4"
    other.write_bytes(b"contenido distinto")
    assert cache.content_key(other, "transcribe", "whisper|es") != k1, "cambia el contenido"


def test_cache_put_get_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("CENTAURO_CACHE", str(tmp_path / "cache"))
    video = tmp_path / "a.mp4"
    video.write_bytes(b"x")
    key = cache.content_key(video, "understand", "m|f=12")

    assert cache.get("understand", key) is None
    cache.put("understand", key, {"resumen": "ok"})
    assert cache.get("understand", key) == {"resumen": "ok"}


def test_cache_prune_removes_old_entries(tmp_path, monkeypatch):
    import os
    import time

    monkeypatch.setenv("CENTAURO_CACHE", str(tmp_path / "cache"))
    cache.put("understand", "k", {"v": 1})
    old = tmp_path / "cache" / "understand" / "k.json"
    past = time.time() - 40 * 86400
    os.utime(old, (past, past))

    assert cache.prune(older_than_days=30) == 1
    assert cache.get("understand", "k") is None


# --- canonicalización de entidades -----------------------------------------

def test_canonical_merges_variants():
    canon = entities.canonical("Logotipo de Telcel")
    assert canon == entities.canonical("logo Telcel") == "logo telcel"
    assert entities.canonical("  TELÉFONO  Móvil ") == entities.canonical("celular")


def test_extract_entities_dedups_within_ad():
    u = {
        "objetos": ["teléfono", "Telefono", "  "],
        "personajes": ["una pareja"],
        "marca_elementos": ["Logotipo de Telcel"],
        "temas": [],
    }
    out = entities.extract_entities(u)
    keys = [(k, c) for k, c, _ in out]
    assert len(keys) == len(set(keys)), "sin duplicados por (kind, canonical)"
    assert ("marca", "logo telcel") in keys
    assert ("obj", "telefono") in keys


# --- store ------------------------------------------------------------------

def test_store_upsert_roundtrip_and_entities(tmp_path):
    con = store.connect(tmp_path / "t.db")
    store.init(con)

    rec = {"id": "A1", "url": "u", "title": "t", "duration": 30.0,
           "transcript": {"text": "hola"}, "understanding": {"resumen": "r"}}
    store.upsert_ad(con, rec, [("marca", "logo telcel", "Logo Telcel")])

    got = store.get_ad(con, "A1")
    assert got["transcript"] == {"text": "hola"}
    assert got["understanding"] == {"resumen": "r"}
    assert store.top_entities(con) == [("logo telcel", 1)]

    # un upsert posterior sin transcript NO debe borrar el ya calculado
    store.upsert_ad(con, {"id": "A1", "url": "u"})
    assert store.get_ad(con, "A1")["transcript"] == {"text": "hola"}


# --- grafo ------------------------------------------------------------------

def test_graph_links_ads_sharing_entities():
    recs = [
        {"id": "A", "title": "a", "understanding": {"marca_elementos": ["Logotipo de Telcel"],
                                                     "objetos": ["teléfono"]}},
        {"id": "B", "title": "b", "understanding": {"marca_elementos": ["logo telcel"],
                                                     "temas": ["conectividad"]}},
    ]
    G = graph.build_graph(recs)
    assert G.has_edge("A", "B"), "comparten 'logo telcel' pese a escribirse distinto"
    assert G.nodes["marca:logo telcel"]["kind"] == "marca"


# --- guardia de calidad del batch -------------------------------------------

def test_looks_degenerate_flags_loops_and_bad_json():
    from discovery import understand

    assert understand.looks_degenerate({"_raw": "texto no json"})
    assert understand.looks_degenerate({"_error": "boom"})
    assert understand.looks_degenerate({"marca_elementos": ["Telcel", "Telcel", "Telcel"]})
    assert understand.looks_degenerate({"objetos": [f"o{i % 2}" for i in range(50)]})
    assert not understand.looks_degenerate(
        {"objetos": ["globo"], "temas": ["viaje"], "marca_elementos": ["Telcel", "logo"]}
    )
