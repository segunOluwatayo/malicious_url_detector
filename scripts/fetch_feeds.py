import requests
import pandas as pd
import pathlib as p
from tranco import Tranco

BASE = p.Path(__file__).parents[1] / "data"
BASE.mkdir(exist_ok=True)

def urlhaus():
    url = "https://urlhaus.abuse.ch/downloads/csv_recent/"
    # grab the whole feed as text
    r = requests.get(url, timeout=30)
    lines = r.text.splitlines()
    # skip comments, split on commas, take field #1
    urls = [
        line.split(",")[1]
        for line in lines
        if line and not line.startswith("#") and "," in line
    ]
    df = pd.DataFrame({"url": urls})
    df["label"] = 1
    return df

def phishtank():
    url = "https://data.phishtank.com/data/online-valid.csv.bz2"
    df = pd.read_csv(url, compression="bz2", usecols=["url"])
    df["label"] = 1
    return df

def tranco(top=1_000_000, date=None):
    t = Tranco(cache=True)
    tranco_list = t.list(date=date)
    domains = tranco_list.top(top)
    df = pd.DataFrame(domains, columns=["domain"])
    df["url"] = "http://" + df["domain"]
    df["label"] = 0
    return df[["url", "label"]]

if __name__ == "__main__":
    feeds = [phishtank(), urlhaus(), tranco()]
    raw = pd.concat(feeds).drop_duplicates("url").reset_index(drop=True)
    out = BASE / "raw.csv"
    raw.to_csv(out, index=False)
    print(f"Saved {out} with {len(raw)} URLs")
