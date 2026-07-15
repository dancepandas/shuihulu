import requests, re, sys
from pathlib import Path

base = 'http://192.168.30.107:8090'
s = requests.Session()
r = s.get(f'{base}/user/login/')
m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text)
csrf = m.group(1) if m else None
s.post(f'{base}/user/login/', data={'email': 'chs9710@163.com', 'password': 'cHENGS1997', 'csrfmiddlewaretoken': csrf})
h = {'X-CSRFToken': s.cookies.get('csrftoken', '')}

all_tasks = []
for page in range(1, 10):
    r = s.get(f'{base}/api/projects/1/tasks?page={page}&page_size=500', headers=h)
    if r.status_code != 200: break
    data = r.json()
    tasks = data if isinstance(data, list) else data.get('tasks', [])
    if not tasks: break
    all_tasks.extend(tasks)
    print(f'  page {page}: {len(all_tasks)} tasks')

unlabeled = [t for t in all_tasks if not t.get('annotations')]

def ts(t):
    p = Path(t['data'].get('image', '')).name.split('_')
    return p[1] if len(p) >= 2 else ''

unlabeled.sort(key=ts, reverse=True)
latest = unlabeled[:1000]
print(f'Latest 1000 unlabeled, downloading...')

img_dir = Path('data/ls_images')
d = 0
for i, t in enumerate(latest):
    fn = Path(t['data'].get('image', '')).name
    if (img_dir / fn).exists():
        d += 1
        continue
    r2 = s.get(f'{base}/api/tasks/{t["id"]}?resolve_uri=true', headers=h)
    if r2.status_code == 200 and len(r2.content) > 1000:
        with open(img_dir / fn, 'wb') as f:
            f.write(r2.content)
        d += 1
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(latest)} done, {d} local')

print(f'Done: {d}/{len(latest)} images')
