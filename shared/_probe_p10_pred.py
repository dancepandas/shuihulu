"""探查 LS project 10 的 labeling config 和 predictions API 可行性。"""
import re, time, json, requests

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
    v = s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=60)
    if v.status_code == 200: break
    time.sleep(3)

# 1. labeling config
proj = s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=60).json()
print("=== labeling_config ===")
print(proj.get("label_config", "")[:1500])

# 2. 未标注任务数
r = s.get(f"{BASE}/api/projects/{PROJECT}/tasks",
          params={"fields":"id,is_labeled,data","page":1,"page_size":500}, headers=h, timeout=120)
tasks = r.json()
unlabeled = [t for t in tasks if not t.get("is_labeled")]
print(f"\n=== 第一页 {len(tasks)} 条, 未标注 {len(unlabeled)} ===")
if unlabeled:
    t = unlabeled[0]
    print("task id:", t["id"])
    print("data:", t["data"])

# 3. 试 predictions API 可行性 (空 result 测试, 仅验证接口响应)
if unlabeled:
    tid = unlabeled[0]["id"]
    test = {"model_version": "probe", "result": []}
    p = s.post(f"{BASE}/api/tasks/{tid}/predictions", json=test,
               headers={**h, "Content-Type": "application/json"}, timeout=60)
    print(f"\n=== POST /tasks/{tid}/predictions (empty) -> HTTP {p.status_code} ===")
    print(p.text[:200] if p.status_code >= 400 else "OK")
    # 删除这个测试 prediction
    d = s.delete(f"{BASE}/api/tasks/{tid}/predictions", headers=h, timeout=60)
    print(f"cleanup delete -> HTTP {d.status_code}")
