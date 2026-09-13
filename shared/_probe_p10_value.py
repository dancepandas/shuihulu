"""细查 result[0]['value'] 的真实形状。"""
import re, json, requests, time
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
r = s.get(f"{BASE}/api/projects/{PROJECT}/tasks",
          params={"fields":"all","page":1,"page_size":50}, headers=h, timeout=120)
r.raise_for_status()
t = [x for x in r.json() if x.get("is_labeled")][0]
a = t["annotations"][0]
print(f"task_id={t['id']}  ann_id={a['id']}  completed_by={a['completed_by']}")
print(f"image path: {t['data']['image']}")
print(f"\nresult count: {len(a['result'])}")
for i, res in enumerate(a["result"]):
    print(f"\n--- result[{i}] ---")
    print(f"keys: {sorted(res.keys())}")
    print(f"type={res.get('type')}  from_name={res.get('from_name')}  to_name={res.get('to_name')}")
    val = res.get("value", {})
    if isinstance(val, dict):
        print(f"value keys: {sorted(val.keys())}")
        print(f"value.rle len={len(val.get('rle',[]))}  first3={val.get('rle',[])[:3]}")
        print(f"value.brushlabels: {val.get('brushlabels')}")
        print(f"value.format: {val.get('format')}")
    else:
        print(f"value (not dict): {val!r}"[:200])
    print(f"original_width={res.get('original_width')}  original_height={res.get('original_height')}")