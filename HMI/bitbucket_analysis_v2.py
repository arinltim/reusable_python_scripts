import time
import requests
import pandas as pd
from datetime import datetime

BB_USERNAME     = "aseal1"
BB_APP_PASSWORD = ""
WORKSPACE       = "horizonspireteam"
CUTOFF          = datetime(2025,1,1)

def list_repos():
    url, params, slugs = f"https://api.bitbucket.org/2.0/repositories/{WORKSPACE}", {"pagelen":100}, []
    while url:
        r = requests.get(url, auth=(BB_USERNAME,BB_APP_PASSWORD), params=params, verify=False)
        r.raise_for_status()
        j = r.json()
        for repo in j["values"]:
            dt_aware = datetime.fromisoformat(repo["updated_on"])
            dt = dt_aware.replace(tzinfo=None)
            if dt > CUTOFF:
                slugs.append(repo["slug"])
        url = j.get("next"); params=None
    print(len(slugs))
    return slugs

def page_values(url, params=None):
    """Generator to yield each page’s `values` list, honoring pagination."""
    while url:
        r = requests.get(url, auth=(BB_USERNAME,BB_APP_PASSWORD), params=params, verify=False)
        r.raise_for_status()
        j = r.json()
        yield j["values"]
        url = j.get("next"); params=None

def analyze_users(user_list):
    repos = list_repos()
    # initialize counters
    stats = {u: {"prs":0, "commits":0, "comments":0} for u in user_list}

    for slug in repos:
        base = f"https://api.bitbucket.org/2.0/repositories/{WORKSPACE}/{slug}"

        # 1) PullRequests
        for page in page_values(f"{base}/pullrequests", {}):
            for pr in page:
                author = pr["author"]["display_name"]
                if author in stats:
                    stats[author]["prs"] += 1

        # 2) Commits
        for page in page_values(f"{base}/commits", {}):
            for c in page:
                au = c["author"].get("user", {}).get("display_name")
                if au in stats:
                    stats[au]["commits"] += 1

        # 3) PR Comments
        # for page in page_values(f"{base}/pullrequests/comments", {"pagelen":100}):
        #     for cm in page:
        #         commenter = cm["user"]["display_name"]
        #         if commenter in stats:
        #             stats[commenter]["comments"] += 1

        # simple throttle
        time.sleep(1)

    # turn into DataFrame
    rows = [
        {"username":u, **stats[u]}
        for u in user_list
    ]
    print(rows)
    return pd.DataFrame(rows)

if __name__=="__main__":
    users = [
        'LMutha', 'Arun Mothe', 'Seva Adari', 'npatel',
        'Bhavana Vankayala', 'pregalla', 'RRavichandran', 'msingla',
        'Geoffrey Link', 'Oyal Fokshner', 'Ranga Nanda',
        'Abhishek Reddy Regalla', 'BhKumarAmma'
    ]
    df = analyze_users(users)
    df.to_excel("bitbucket_user_report.xlsx", index=False)
    print("✅ Done")
