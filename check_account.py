import json, sqlite3
conn = sqlite3.connect('/app/data/accounts.db')
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT * FROM accounts LIMIT 3").fetchall()
for r in rows:
    d = dict(r)
    # Hide sensitive values
    for k in list(d.keys()):
        if 'token' in k.lower() or 'cookie' in k.lower():
            v = d[k]
            if v and isinstance(v, str):
                d[k] = v[:20] + '...' if len(v) > 20 else v
    print(json.dumps(d, indent=2, default=str))
conn.close()
