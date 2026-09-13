"""
从老 LS 服务器 (192.168.30.107:8090) 下载 p7o / p9 笔刷任务对应的原图
- p7o: data/ls_export_p7o 中的 task ID → 项目7 → data/ls_images_p7o/{tid}.jpg
- p9 : data/ls_export_p9  中的 task ID → 项目9 → data/ls_images_p9/{tid}.jpg
"""
import os, re, time
import requests

BASE = "http://192.168.30.107:8090"
ROOT = r"D:\chengs\9.project\shuihulu"

JOBS = [
    (os.path.join(ROOT, "data", "ls_export_p7o"), os.path.join(ROOT, "data", "ls_images_p7o")),
    (os.path.join(ROOT, "data", "ls_export_p9"),  os.path.join(ROOT, "data", "ls_images_p9")),
]

s = requests.Session()
r = s.get(f"{BASE}/user/login/")
m = re.search(r'csrfmiddlewaretoken" value="([^"]+)"', r.text)
s.post(f"{BASE}/user/login/", data={
    "csrfmiddlewaretoken": m.group(1),
    "email": "chs9710@163.com",
    "password": "cHENGS1997",
}, headers={"Referer": f"{BASE}/user/login/"})
print("logged in")

for mask_dir, img_dir in JOBS:
    os.makedirs(img_dir, exist_ok=True)
    tids = sorted({int(re.match(r"task-(\d+)-", f).group(1))
                   for f in os.listdir(mask_dir) if re.match(r"task-(\d+)-", f)})
    print(f"\n{os.path.basename(mask_dir)}: {len(tids)} tasks")
    ok = skip = fail = 0
    for i, tid in enumerate(tids):
        out = os.path.join(img_dir, f"{tid}.jpg")
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            skip += 1
            continue
        try:
            t = s.get(f"{BASE}/api/tasks/{tid}/", timeout=30).json()
            url = BASE + t["data"]["image"]
            r = s.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 10000:
                with open(out, "wb") as f:
                    f.write(r.content)
                ok += 1
            else:
                fail += 1
                print(f"  task {tid}: bad response {r.status_code} len={len(r.content)}")
        except Exception as e:
            fail += 1
            print(f"  task {tid}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tids)} (ok={ok} skip={skip} fail={fail})")
    print(f"done: ok={ok} skip={skip} fail={fail}")
