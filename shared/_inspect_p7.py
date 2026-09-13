"""检查 project 7 里某个 task 的 prediction 真实结构"""
import json, re, requests, base64, numpy as np

BASE = "http://localhost:8080"
EMAIL, PWD = "chs9710@163.com", "Chengs1997"

s = requests.Session(); s.trust_env = False
g = s.get(f"{BASE}/user/login/", timeout=30)
m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
s.post(f"{BASE}/user/login/", data={"email": EMAIL, "password": PWD, "csrfmiddlewaretoken": m.group(1)},
       headers={"Referer": f"{BASE}/user/login/"}, timeout=30)
h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}

# 取第一个 task (带 predictions)
r = s.get(f"{BASE}/api/projects/7/tasks/?fields=predictions&per_page=2", headers=h, timeout=30)
data = r.json()
print("HTTP", r.status_code, "tasks:", len(data))
for t in data:
    tid = t["id"]
    preds = t.get("predictions", [])
    print(f"\n=== task {tid}  #predictions={len(preds)} ===")
    if preds:
        pr = preds[0]["result"]
        print(f"  result items: {len(pr)}")
        item = pr[0]
        print(f"  type: {item['type']}")
        print(f"  from_name: {item.get('from_name')}  to_name: {item.get('to_name')}")
        v = item["value"]
        print(f"  value keys: {list(v.keys())}")
        print(f"  format: {v.get('format')}")
        print(f"  original_width: {v.get('original_width')}  original_height: {v.get('original_height')}")
        rle = v.get("rle", [])
        print(f"  rle len: {len(rle)}  first 20: {rle[:20]}")
        print(f"  rle sum: {sum(rle)}")
        # 尝试用我的 mask_to_rle 解码回去看对不对
        flat_len = v.get("original_width",0)*v.get("original_height",0)
        print(f"  expected flat len: {flat_len}")
        print(f"  brushlabels: {v.get('brushlabels')}")
    break
