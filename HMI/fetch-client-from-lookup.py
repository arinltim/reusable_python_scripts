import pandas as pd

# 1. Load both Excels
df_source = pd.read_excel('13_associates_clients - Copy.xlsx')
df_lookup = pd.read_excel('13_associates_data_with_client_3months.xlsx')

# 2. Build mapping dicts from lookup
#    - issue_key_map: maps Issue key -> Client
#    - summary_map:   maps Summary   -> Client
issue_key_map = dict(zip(df_lookup['Issue key'], df_lookup['Client']))
summary_map   = dict(zip(df_lookup['Summary'], df_lookup['Client']))

# 3. Define lookup function with fallback
def fetch_client(row):
    # Try direct issue-key lookup
    client = issue_key_map.get(row['Issue key'], None)
    # If not found or blank/NaN, fallback to summary lookup
    if client is None or (isinstance(client, float) and pd.isna(client)) or client == '':
        client = summary_map.get(row.get('Summary'), None)
    return client

# 4. Apply to source DataFrame
#    This will create or overwrite the 'Client' column
if 'Summary' not in df_source.columns:
    raise KeyError("Source DataFrame missing 'Summary' column for fallback lookup.")

df_source['Client'] = df_source.apply(fetch_client, axis=1)

# 5. Save the updated Excel
output_path = '13_associates_clients.xlsx'
df_source.to_excel(output_path, index=False)
print(f"Saved updated file to {output_path}")
