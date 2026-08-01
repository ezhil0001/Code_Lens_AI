import json, urllib.request, urllib.error, uuid, time

B = 'http://127.0.0.1:8001'
out = {}

req = urllib.request.Request(B+'/api/v1/auth/login?force_login=true',
    data=json.dumps({'email':'e2e@codelens.ai','password':'E2eTest!2026'}).encode(),
    headers={'Content-Type':'application/json'})
login = json.load(urllib.request.urlopen(req, timeout=30))
tok = login['access_token']; uid = login['user']['id']
open('/tmp/token.txt','w').write(tok); open('/tmp/uid.txt','w').write(uid)
out['login'] = 'OK uid=' + uid

session_id = str(uuid.uuid4()); open('/tmp/session_id.txt','w').write(session_id)
body = json.dumps({'query':'How does the authentication flow work in this codebase?',
                   'session_id':session_id, 'user_id':uid}).encode()
req = urllib.request.Request(B+'/api/v2/chat/stream', data=body,
    headers={'Content-Type':'application/json','Authorization':'Bearer '+tok,'Accept':'text/event-stream'})
t0 = time.time(); events = {}; raw = []; final = {}
try:
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            line = line.decode(errors='replace').rstrip('\n'); raw.append(line)
            if line.startswith('data:'):
                try:
                    payload = json.loads(line[5:].strip())
                    t = payload.get('type', '?')
                    events[t] = events.get(t, 0) + 1
                    if t in ('done', 'error', 'metadata', 'sources', 'routing'):
                        final[t] = payload.get('data', payload)
                except Exception:
                    pass
except urllib.error.HTTPError as e:
    out['stream_http_error'] = [e.code, e.read().decode()[:500]]
out['elapsed'] = round(time.time()-t0,1)
out['events'] = events
out['final'] = final
# reconstruct answer from token events
answer = ''
for line in raw:
    if line.startswith('data:'):
        try:
            p = json.loads(line[5:].strip())
            if p.get('type') == 'token':
                answer += p.get('data', {}).get('content', '')
        except Exception:
            pass
out['answer_preview'] = answer[:600]
out['answer_len'] = len(answer)
out['trace_id'] = (final.get('done') or {}).get('trace_id')
open('/tmp/sse_raw.txt','w').write('\n'.join(raw))
open('/tmp/e2e_out.json','w').write(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:3000])
