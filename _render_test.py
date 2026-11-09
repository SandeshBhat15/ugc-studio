import json, time, sys, urllib.request

BASE = 'http://127.0.0.1:8756'
PID = 'owl_coffee_4pm_slump_534858'
SCENES = [int(x) for x in (sys.argv[1:] or ['1'])]


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data,
                               headers={'Content-Type': 'application/json'},
                               method=method)
    with urllib.request.urlopen(r, timeout=180) as x:
        return json.loads(x.read().decode())


def get():
    return json.loads(urllib.request.urlopen(f'{BASE}/api/projects/{PID}',
                                             timeout=30).read())


# approve any stills that are waiting
for s in SCENES:
    entry = get()['stage02']['scenes'].get(str(s)) or {}
    if entry.get('status') == 'awaiting_approval':
        print('approve', s, req('POST', f'/api/projects/{PID}/stills/approve',
                                {'scene': s}).get('status'))

t0 = time.time()
print(req('POST', f'/api/projects/{PID}/videos', {'scenes': SCENES}))
last = None
for i in range(300):
    time.sleep(20)
    p = get()
    s3 = p['stage03']
    st = {k: (v or {}).get('status') for k, v in (s3.get('scenes') or {}).items()}
    if st != last:
        print(f'{int(time.time()-t0):5d}s', s3.get('status'), st, flush=True)
        last = st
    if all(st.get(str(s)) in ('completed', 'failed', 'needs_attention')
           for s in SCENES):
        break
print('elapsed', round(time.time() - t0, 1), 's')
print(json.dumps({k: {kk: (v or {}).get(kk) for kk in
                      ('status', 'file', 'duration', 'width', 'height', 'error', 'qc')}
                  for k, v in (get()['stage03'].get('scenes') or {}).items()
                  if int(k) in SCENES}, indent=1)[:2200])
