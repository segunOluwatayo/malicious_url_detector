# scripts/build_bloom.py
from bloom_filter2 import BloomFilter
import requests, pickle, tldextract
from feeds import DATA              # ← keep the same import style everywhere
from allowlist import GOOD          # ← NEW: bring in the allow-list

def main() -> None:
    bad = set()

    # --------- live feeds ------------------------------------------------ #
    bad.update(
        l.strip() for l in
        requests.get("https://urlhaus.abuse.ch/downloads/text_recent/").text.splitlines()
        if l and not l.startswith("#")
    )
    bad.update(
        requests.get(
            "https://raw.githubusercontent.com/spamhaus/dblmaster/official-release/dbl.txt"
        ).text.splitlines()
    )

    # --------- keep only registered-domain, strip allow-listed ----------- #
    bad = {
        tldextract.extract(d).top_domain_under_public_suffix
        for d in bad if d
    }
    bad -= GOOD                                       # ← NEW

    # --------- build & save Bloom filter ------------------------------- #
    bf = BloomFilter(max_elements=len(bad), error_rate=0.01)
    for d in bad:
        bf.add(d)

    with open(DATA / "bad_domains.bloom", "wb") as f:
        pickle.dump(bf, f)

    print(f"bad_domains.bloom saved  ({len(bad):,} domains)")

if __name__ == "__main__":
    main()
