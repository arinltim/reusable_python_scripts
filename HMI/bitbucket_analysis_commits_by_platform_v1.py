import requests
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone

# ─── CONFIG ────────────────────────────────────────────────────────────────
BB_USERNAME     = "abose4"
BB_APP_PASSWORD = ""
WORKSPACE       = "horizonspireteam"

# the users you want to report on (must match the display_name in Bitbucket)
USERS = [
    'LMutha', 'Arun Mothe', 'Seva Adari', 'npatel',
    'Bhavana Vankayala', 'pregalla', 'RRavichandran', 'msingla',
    'Geoffrey Link', 'Oyal Fokshner', 'Ranga Nanda',
    'Abhishek Reddy Regalla', 'BhKumarAmma'
]

# cut‑off date (inclusive)
CUTOFF = datetime(2025, 1, 1, tzinfo=timezone.utc)

# ─── HELPERS ────────────────────────────────────────────────────────────────
def page_values(url, params=None):
    """Yield all `values` entries across paginated Bitbucket 2.0 responses."""
    while url:
        resp = requests.get(url,
                            auth=(BB_USERNAME, BB_APP_PASSWORD),
                            params=params)
        resp.raise_for_status()
        data = resp.json()
        yield from data.get("values", [])
        url = data.get("next")
        params = None

# ─── 1) FETCH REPO→LANGUAGE MAPPING (post-CUTOFF) ───────────────────────────
repo_to_lang = {}
for repo in page_values(
        f"https://api.bitbucket.org/2.0/repositories/{WORKSPACE}",
        params={"pagelen": 100}
):
    updated_on = repo.get("updated_on")
    if not updated_on:
        continue
    # parse ISO timestamp including timezone
    dt = datetime.fromisoformat(updated_on)
    if dt >= CUTOFF:
        slug = repo["slug"]
        lang = repo.get("language") or "unknown"
        repo_to_lang[slug] = lang

print(f"Using {len(repo_to_lang)} repos updated on/after {CUTOFF.date()}")

# ─── 2) COUNT COMMITS BY USER & LANGUAGE ────────────────────────────────────
user_lang_counts = {u: defaultdict(int) for u in USERS}

for slug, lang in repo_to_lang.items():
    commits_endpoint = (
        f"https://api.bitbucket.org/2.0/repositories/"
        f"{WORKSPACE}/{slug}/commits"
    )
    for commit in page_values(commits_endpoint, params={"pagelen": 100}):
        author = commit.get("author", {}) \
            .get("user", {}) \
            .get("display_name")
        if author in user_lang_counts:
            user_lang_counts[author][lang] += 1

# ─── 3) BUILD & SAVE DATAFRAME ─────────────────────────────────────────────
rows = []
for user, counts in user_lang_counts.items():
    row = {"username": user}
    row.update(counts)
    rows.append(row)

df = pd.DataFrame(rows).fillna(0).set_index("username")
df.to_excel("commit_by_language_since_2025-01-01.xlsx")

print("✅ Done — see commit_by_language_since_2025-01-01.xlsx")
