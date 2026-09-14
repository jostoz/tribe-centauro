"""Descubrimiento y descarga de anuncios desde YouTube (yt-dlp, gratis)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List

import imageio_ffmpeg
import yt_dlp

DEFAULT_OUT = Path("data/ads")


def _ensure_ffmpeg_dir() -> str:
    """yt-dlp busca un binario llamado ffmpeg(.exe); el de imageio-ffmpeg tiene
    otro nombre. Lo copiamos una vez a un dir estable con el nombre esperado."""
    dst_dir = Path(tempfile.gettempdir()) / "centauro_ffmpeg"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / ("ffmpeg.exe" if __import__("os").name == "nt" else "ffmpeg")
    if not dst.exists():
        try:
            shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), dst)
        except (OSError, shutil.SameFileError):
            # carrera con otro hilo de descarga: si ya quedó copiado, seguimos
            if not dst.exists():
                raise
    return str(dst_dir)


def _ydl_download_opts(outdir: Path) -> dict:
    return {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "ffmpeg_location": _ensure_ffmpeg_dir(),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ignoreerrors": True,
    }


def _record(info: dict, outdir: Path) -> dict:
    return {
        "id": info["id"],
        "source": "youtube",
        "url": info.get("webpage_url"),
        "title": info.get("title"),
        "duration": info.get("duration"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "video_path": str(outdir / f"{info['id']}.mp4"),
    }


def fetch_stats(urls: Iterable[str], workers: int = 4) -> List[dict]:
    """Métricas **públicas** de YouTube (vistas, likes, comentarios) sin descargar el video.

    Son un proxy débil de resultado: mezclan pauta pagada con orgánico y, en re-subidas de
    terceros, no son métricas de la marca. Sirven para ordenar dentro de un mismo canal,
    no para calibrar contra resultados de campaña.
    """
    from concurrent.futures import ThreadPoolExecutor

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
    }

    def one(url: str):
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            return None
        return {
            "id": info["id"],
            "url": info.get("webpage_url") or url,
            "title": info.get("title"),
            "duration": info.get("duration"),
            "channel": info.get("channel") or info.get("uploader"),
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
        }

    urls = list(urls)
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return [r for r in pool.map(one, urls) if r]


def download(urls: Iterable[str], outdir: Path | str = DEFAULT_OUT) -> List[dict]:
    """Descarga videos y devuelve un registro por anuncio."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records: List[dict] = []
    with yt_dlp.YoutubeDL(_ydl_download_opts(outdir)) as ydl:
        for url in urls:
            info = ydl.extract_info(url, download=True)
            if info is None:
                continue
            if "entries" in info:  # playlist
                for e in info["entries"]:
                    if e:
                        records.append(_record(e, outdir))
            else:
                records.append(_record(info, outdir))
    return records


def download_many(urls: Iterable[str], outdir: Path | str = DEFAULT_OUT, workers: int = 4) -> List[dict]:
    """Descarga en paralelo (I/O de red) y devuelve un registro por anuncio.

    Descargar es I/O-bound: solaparlo con la GPU evita que el worker espere la red.
    Filas ordenadas por id para que la corrida sea reproducible.
    """
    from concurrent.futures import ThreadPoolExecutor

    urls = list(urls)
    records: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for part in pool.map(lambda u: download([u], outdir), urls):
            records.extend(part)
    return sorted(records, key=lambda r: r["id"])


def _flat_urls(target: str, n: int) -> List[str]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "playlistend": n,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(target, download=False)
    entries = (info or {}).get("entries", []) or []
    out = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if vid:
            out.append(f"https://www.youtube.com/watch?v={vid}")
    return out[:n]


def search_urls(query: str, n: int = 10) -> List[str]:
    """URLs de los primeros ``n`` resultados de búsqueda en YouTube."""
    return _flat_urls(f"ytsearch{n}:{query}", n)


def channel_urls(handle: str, n: int = 20) -> List[str]:
    """URLs de los últimos ``n`` videos de un canal (handle tipo ``@Telcel``)."""
    handle = handle if handle.startswith(("@", "http")) else f"@{handle}"
    target = handle if handle.startswith("http") else f"https://www.youtube.com/{handle}/videos"
    return _flat_urls(target, n)
