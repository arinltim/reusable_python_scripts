import pandas as pd
import re

# 1. Load both files
main_df   = pd.read_excel("13_associates_data_with_client.xlsx", header=0).fillna('')
lookup_df = pd.read_excel("Book1.xlsx",                    header=0).fillna('')

# 2. Helper to normalize text: lowercase, strip, remove non-alphanum
def normalize(s):
    if not isinstance(s, str):
        return ""
    collapsed = re.sub(r'\s+', ' ', s)
    s = collapsed.strip().lower()
    return s

# 3. Add a clean key column to both
main_df["_key"]   = main_df["Summary"].apply(normalize)
lookup_df["_key"] = lookup_df["Summary"].apply(normalize)

# 4. Build lookup dict on clean key ➔ extracted client
lookup_dict = dict(zip(
    lookup_df["_key"],
    lookup_df["Extracted Client Name"].astype(str).str.strip()
))

# 5. Populate Client_New: prefer migrated, else lookup by clean key
def get_client_new(row):
    mig = str(row["Custom field (Client (migrated))"]).strip()
    if mig and str(mig).strip() != '':
        return mig

    candidate = lookup_dict.get(row["_key"], "").strip()
    client_str = re.sub(r'\s+', ' ', row.get("Client", "")).strip()
    return candidate if candidate else client_str

main_df["Client_New"] = main_df.apply(get_client_new, axis=1)

# 6. Drop helper column and save
main_df.drop(columns=["_key"], inplace=True)
main_df.to_excel("13_associates_data_with_client_new.xlsx", index=False)

print("✅ Done — ‘13_associates_data_with_client_new.xlsx’ generated!")
