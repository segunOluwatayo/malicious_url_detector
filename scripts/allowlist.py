# scripts/allowlist.py
"""
Provide GOOD – a set of ~1 000 000 high-reputation domains.

On the first call we try to download the public Tranco Top-1M CSV.
If that fails (bad gateway, HTML, not gzipped, …) we fall back to the
Tranco Python API.  The result is cached in data/allowlist.txt.
"""
from pathlib import Path
import time, requests, gzip, io, sys

DATA  = Path(__file__).with_suffix("").parent / "data"
CACHE = DATA / "allowlist.txt"
URL   = "https://tranco-list.eu/top-1m.csv.gz"

def _download_via_csv() -> list[str]:
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()

    try:
        raw = gzip.decompress(resp.content)        # may raise OSError
    except OSError:
        # not a valid gzip – maybe the site returned an error page
        raise RuntimeError("CSV not gzipped")

    text = raw.decode(errors="replace")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("rank,domain"):
        raise RuntimeError("CSV format unexpected")

    return [line.split(",")[1].strip() for line in lines[1:]]

def _download_via_api() -> list[str]:
    print("Falling back to Tranco API (this may take ~1 min)…", file=sys.stderr)
    from tranco import Tranco
    return Tranco(cache=True).list().top(1_000_000)

def _ensure_cache() -> None:
    DATA.mkdir(exist_ok=True)
    try:
        domains = _download_via_csv()
    except Exception as e:
        print(f"Top-1M CSV download failed: {e}", file=sys.stderr)
        domains = _download_via_api()

    CACHE.write_text("\n".join(domains))
    print(f"Wrote allow-list with {len(domains):,} domains → {CACHE}")

def load() -> set[str]:
    if not CACHE.exists() or time.time() - CACHE.stat().st_mtime > 86400:
        _ensure_cache()
    return set(CACHE.read_text().splitlines())

GOOD = load()
