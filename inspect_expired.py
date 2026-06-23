import json
data = json.load(open('/home/ubuntu/ts/grok2api/data/token.json'))
ss = data.get('ssoSuper', [])
expired_0 = [t for t in ss if t.get('status') == 'expired' and t.get('quota', 0) <= 0]
print(f'Total expired+no_quota: {len(expired_0)}')
print()

# Show summary of all fields
field_summary = {}
for t in expired_0:
    for k, v in t.items():
        if k == 'token':
            continue
        if k not in field_summary:
            field_summary[k] = []
        if isinstance(v, (int, float, type(None))):
            field_summary[k].append(v)
        else:
            field_summary[k].append(str(v)[:80])

for f, vals in sorted(field_summary.items()):
    unique = set(str(v)[:50] for v in vals)
    if len(unique) <= 5:
        print(f'{f}: {unique}')
    else:
        print(f'{f}: {len(unique)} unique values, sample: {list(unique)[:3]}')

# Sample first token
print()
print("=== Sample token (first) ===")
t = expired_0[0]
for k, v in sorted(t.items()):
    if k == 'token':
        v = v[:20] + '...'
    print(f'  {k}: {v}')
