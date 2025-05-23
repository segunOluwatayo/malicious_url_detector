from pathlib import Path
import pandas as pd, requests, bz2, io
from tranco import Tranco
from allowlist import GOOD          

DATA = Path(__file__).with_suffix("").parent / "data"
DATA.mkdir(exist_ok=True)

# -------------- helpers ------------------------------------------------ #
def _get_urlhaus() -> pd.DataFrame:
    txt = requests.get("https://urlhaus.abuse.ch/downloads/csv_recent/",
                       timeout=30).text.splitlines()
    urls = [l.split(",")[1] for l in txt
            if l and not l.startswith("#") and "," in l]  
    return pd.DataFrame({"url": urls, "label": 1})

def _get_phishtank() -> pd.DataFrame:
    raw = requests.get("https://data.phishtank.com/data/online-valid.csv.bz2",
                       timeout=60).content
    csv = bz2.decompress(raw)
    return pd.read_csv(io.BytesIO(csv), usecols=["url"]).assign(label=1)

def _get_openphish() -> pd.DataFrame:
    txt = requests.get("https://openphish.com/feed.txt",
                       timeout=30).text.splitlines()
    return pd.DataFrame({"url": txt, "label": 1})

def _get_tranco(top: int = 100_000) -> pd.DataFrame:
    client  = Tranco(cache=True)
    domains = client.list().top(top)
    return pd.DataFrame({"url": [f"https://{d}/" for d in domains], "label": 0})

# -------------- public API --------------------------------------------- #
def get_dataframe() -> pd.DataFrame:
    df_bad = pd.concat(
        [_get_urlhaus(), _get_phishtank(), _get_openphish()],
        ignore_index=True,
    )

    # Strip any “bad” sample whose domain is on our reputably-good allow-list
    dom     = df_bad.url.str.extract(r"https?://([^/]+)/", expand=False)
    df_bad  = df_bad[~dom.isin(GOOD)]

    df = pd.concat([df_bad, _get_tranco()], ignore_index=True).drop_duplicates("url")
    return df

# ----------------------------------------------------------------------- #
if __name__ == "__main__":            
    df  = get_dataframe()
    dst = DATA / "raw.csv"
    df.to_csv(dst, index=False)
    print(f"Wrote {dst}  ({len(df):,} rows)")
