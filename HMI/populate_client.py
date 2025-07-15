import re
import pandas as pd

# 1. Load
df = pd.read_excel("13_associates_data.xlsx")

# 2. Define extractor
def extract_client(summary: str) -> str:
    if not isinstance(summary, str) or not summary.strip():
        return ""
    s = summary.strip()
    # 2a. Strip leading "CLONE -"
    s = re.sub(r'(?i)^\s*CLONE\s*-\s*', '', s)
    # 2b. Pipe logic: if there's a '|', the first segment is the client
    if "|" in s:
        return s.split("|", 1)[0].strip()
    # 2c. Colon logic
    if ":" in s:
        pre, post = [p.strip() for p in s.split(":", 1)]
        s = post if (" " in pre) else pre
    # 2d. Hyphen/underscore splitting
    s = s.split("-", 1)[0]
    s = s.split("_", 1)[0]
    return s.strip()

# 3. Prepare migrated values
mig_col = "Custom field (Client (migrated))"
df["_mig_clean"] = df[mig_col].fillna("").astype(str).str.strip()

# 4. Create Client column
df["Client"] = df["_mig_clean"].copy()
blank_mask = df["Client"] == ""
df.loc[blank_mask, "Client"] = df.loc[blank_mask, "Summary"].apply(extract_client)

# 5. (Optional) turn any remaining empty strings into <NA>
df["Client"].replace("", pd.NA, inplace=True)

# 6. Reorder so Client is first
cols = ["Client"] + [c for c in df.columns if c not in ("Client","_mig_clean")]
df = df[cols]

# 7. Save
df.to_excel("13_associates_data_with_client.xlsx", index=False)

print("✅ Done — “output_with_client.xlsx” has your new Client column up front.")
