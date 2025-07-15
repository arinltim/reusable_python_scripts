import os
import time
from datetime import datetime, timezone
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────
BB_USERNAME     = "abose4"
BB_APP_PASSWORD = ""
WORKSPACE       = "horizonspireteam"
REPO_CUTOFF     = datetime(2025, 1, 1, tzinfo=timezone.utc)
MAX_THREADS     = 4
# ───────────────────────────────────────────────────────────────────

API_BASE = "https://api.bitbucket.org/2.0/repositories"

class Diffstat429(Exception):
    """Signal first 429 on diffstat; bail this repo."""
    pass

def page_values(url, params=None, bail_on_429=False, max_retries=3):
    """
    Generator over all items in paginated `values`.
    If bail_on_429==True, first 429 will immediately raise Diffstat429.
    Otherwise, it will back off & retry up to max_retries, then give up that page.
    """
    while url:
        for attempt in range(max_retries):
            r = requests.get(url,
                             auth=(BB_USERNAME, BB_APP_PASSWORD),
                             params=params, verify=False)
            if r.status_code == 429:
                if bail_on_429:
                    raise Diffstat429(f"429 on {url}")
                wait = 2 ** attempt
                print(f"[429] {url} → sleeping {wait}s then retry")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        else:
            # too many failures → drop out of this generator
            return

        data = r.json()
        for v in data.get("values", []):
            yield v

        url = data.get("next")
        params = None
        time.sleep(0.1)

def list_recent_repos():
    """Return all repo slugs last updated after REPO_CUTOFF."""
    slugs = []
    url = f"{API_BASE}/{WORKSPACE}"
    for repo in page_values(url, {"pagelen": 100}):
        upd = repo.get("updated_on")
        if not upd:
            continue
        dt = datetime.fromisoformat(upd.rstrip("Z")).astimezone(timezone.utc)
        if dt > REPO_CUTOFF:
            slugs.append(repo["slug"])
    print(f"→ {len(slugs)} repos updated after {REPO_CUTOFF.date()}")
    return slugs

def get_filtered_commits(user, slug):
    """Yield only SHAs for commits by `user` after REPO_CUTOFF."""
    url = f"{API_BASE}/{WORKSPACE}/{slug}/commits"
    q = (
        f'author.username="{user}" '
        f'AND date>"{REPO_CUTOFF.isoformat()}"'
    )
    params = {"q": q, "fields": "values.hash,next"}
    for page in page_values(url, params):
        sha = page.get("hash")
        if sha:
            yield sha

def count_extensions_for_commit(slug, sha, ext_counts, seen_shas):
    """
    Fetch diffstat for one SHA, tally file extensions once.
    On first 429, bail out of this repo immediately by raising Diffstat429.
    """
    if sha in seen_shas:
        return
    seen_shas.add(sha)

    diff_url = f"{API_BASE}/{WORKSPACE}/{slug}/diffstat/{sha}"
    # bail_on_429=True here means first 429 will raise Diffstat429
    for diff in page_values(diff_url, bail_on_429=True):
        new, old = diff.get("new") or {}, diff.get("old") or {}
        path = new.get("path") or old.get("path")
        if path:
            ext = os.path.splitext(path)[1].lower() or "<no_ext>"
            ext_counts[ext] += 1

def analyze_user(user, repos, seen_shas):
    """Return (user, {ext: count, ...}) for that user across repos."""
    ext_counts = defaultdict(int)

    for slug in repos:
        try:
            for sha in get_filtered_commits(user, slug):
                count_extensions_for_commit(slug, sha, ext_counts, seen_shas)
        except Diffstat429 as e:
            print(f"⏭  {user!r} @ {slug!r}: first 429 → skipping rest of diffstats for this repo")
            continue

    return user, ext_counts

def main():
    users = [
        'LMutha','Arun Mothe','Seva Adari','npatel','Bhavana Vankayala',
        'pregalla','RRavichandran','msingla','Geoffrey Link','Oyal Fokshner',
        'Ranga Nanda','Abhishek Reddy Regalla','BhKumarAmma'
    ]
    repos = list_recent_repos()
    seen_shas = set()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as pool:
        futures = {
            pool.submit(analyze_user, u, repos, seen_shas): u
            for u in users
        }
        for fut in as_completed(futures):
            user, counts = fut.result()
            row = {"username": user, **counts}
            results.append(row)
            print(f"→ done {user!r}")

    df = pd.DataFrame(results).fillna(0)
    df.to_excel("bitbucket_commits_by_ext.xlsx", index=False)
    print("✅ Done — bitbucket_commits_by_ext.xlsx written.")

if __name__ == "__main__":
    main()
