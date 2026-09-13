"""解码 DB 里实际存储的 RLE, 证明前端能渲染"""
import json, re, requests
import numpy as np
from label_studio_sdk.converter.brush import decode_rle

BASE = "http://localhost:8080"
s = requests.Session(); s.trust_env = False
g = s.get(f"{BASE}/user/login/", timeout=30)
m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
s.post(f"{BASE}/user/login/", data={"email": "chs9710@163.com", "password": "Chengs1997",
       "csrfmiddlewaretoken": m.group(1)}, headers={"Referer": f"{BASE}/user/login/"}, timeout=30)
h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}

t = s.get(f"{BASE}/api/projects/7/tasks/?fields=predictions&per_page=1", headers=h, timeout=30).json()[0]
item = t["predictions"][0]["result"][0]
v = item["value"]
W, H = v["original_width"], v["original_height"]
rle = v["rle"]
print(f"W={W} H={H}  rle bytes: max={max(rle)} min={min(rle)}  all<=255:{all(0<=x<=255 for x in rle)}")

decoded = decode_rle(rle)               # 1d, len 应 = H*W*4
print(f"decoded len={len(decoded)}  expected={H*W*4}  match={len(decoded)==H*W*4}")
img4 = np.reshape(decoded, [H, W, 4])
alpha = img4[:, :, 3]
fg = int((alpha > 0).sum())
print(f"foreground pixels: {fg}  ({100*fg/(H*W):.2f}% of image)")
print("OK — mask decodes to valid H×W with foreground" if fg > 0 else "FAIL — no foreground")
