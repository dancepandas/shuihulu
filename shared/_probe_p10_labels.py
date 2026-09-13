"""统计 project 10 所有笔刷标签名（一张任务里出现什么标签、跨任务 union）。"""
import re, requests, time
from collections import Counter
BASE = "http://192.168.30.107:8090"
EMAIL, PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
s = requests.Session(); s.trust_env = False
for _ in range(8):
    g = s.get(f"{BASE}/user/login/", timeout=60)
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text).group(1)
    s.post(f"{BASE}/user/login/",
           data={"email": EMAIL, "password": PWD, "csrfmiddlewaretoken": csrf},
           headers={"Referer": f"{BASE}/user/login/"}, allow_redirects=True, timeout=60)
    h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
    if s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=60).status_code == 200: break
    time.sleep(3)

# 分页拉所有 labeled 任务，只看 annotation result 的 brushlabels（不取 rle 数据，省流量）
labels_per_task = Counter()
labels_union = Counter()
total_labeled = 0
total_results = 0
page = 1
while True:
    r = s.get(f"{BASE}/api/projects/{PROJECT}/tasks",
              params={"fields": "id,is_labeled,annotations", "page": page, "page_size": 500},
              headers=h, timeout=120)
    r.raise_for_status()
    batch = r.json()
    if not batch: break
    for t in batch:
        if not t.get("is_labeled"): continue
        total_labeled += 1
        seen = set()
        for a in t.get("annotations", []):
            for res in a.get("result", []):
                if res.get("type") != "brushlabels": continue
                for lab in res.get("value", {}).get("brushlabels", []):
                    labels_union[lab] += 1
                    seen.add(lab)
                    total_results += 1
        for lab in seen:
            labels_per_task[lab] += 1
    if len(batch) < 500: break
    page += 1

print(f"total labeled tasks: {total_labeled}")
print(f"total brush result entries: {total_results}")
print("\n=== labels union (出现次数) ===")
for k, v in labels_union.most_common():
    print(f"  {k:30s}  {v}")
print("\n=== labels per task (含此标签的任务数) ===")
for k, v in labels_per_task.most_common():
    print(f"  {k:30s}  {v}")