"""统计 LS project 10 的 completed (is_labeled) 任务数。"""
import re, time
import requests

BASE = "http://192.168.30.107:8090"
EMAIL, PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10

s = requests.Session()
s.trust_env = False
for i in range(8):
    g = s.get(f"{BASE}/user/login/", timeout=60)
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
    csrf = m.group(1) if m else ""
    s.post(f"{BASE}/user/login/",
           data={"email": EMAIL, "password": PWD, "csrfmiddlewaretoken": csrf},
           headers={"Referer": f"{BASE}/user/login/"}, allow_redirects=True, timeout=60)
    h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
    v = s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=60)
    if v.status_code == 200:
        break
    time.sleep(3)
else:
    raise SystemExit("login failed")

proj = v.json()
print("=== project 10 info ===")
print(f"  title:        {proj.get('title')}")
print(f"  task_count:               {proj.get('task_count')}")
print(f"  finished_task_count:      {proj.get('finished_task_count')}")
print(f"  total_annotations_count:  {proj.get('total_annotations_count')}")
print(f"  total_predictions_count:  {proj.get('total_predictions_count')}")

# 分页计数 is_labeled
print("\n=== paging tasks (page_size=500) ===")
total = done = 0
page = 1
while True:
    r = s.get(f"{BASE}/api/projects/{PROJECT}/tasks",
              params={"fields": "id,is_labeled,completed_at,annotations_count",
                      "page": page, "page_size": 500},
              headers=h, timeout=120)
    if r.status_code != 200:
        print(f"  page {page}: HTTP {r.status_code}")
        break
    j = r.json()
    if not j:
        break
    total += len(j)
    done += sum(1 for t in j if t.get("is_labeled"))
    if len(j) < 500:
        break
    page += 1

print(f"  total tasks (counted via API): {total}")
print(f"  is_labeled == true:            {done}")
print(f"  not labeled:                   {total - done}")