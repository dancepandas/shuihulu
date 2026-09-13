"""探查 project 10 的任务结构和笔刷 PNG 输出文件名规则。"""
import re, os, json, time
import requests
from label_studio_sdk.converter import brush as ls_brush

BASE = "http://192.168.30.107:8090"
EMAIL, PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
OUT_PROBE = r"D:\chengs\9.project\shuihulu\data\ls_export_p10_probe"

# --- 登录（沿用 _probe_ls.py 思路） ---
s = requests.Session(); s.trust_env = False
for i in range(8):
    g = s.get(f"{BASE}/user/login/", timeout=60)
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text).group(1)
    s.post(f"{BASE}/user/login/",
           data={"email": EMAIL, "password": PWD, "csrfmiddlewaretoken": csrf},
           headers={"Referer": f"{BASE}/user/login/"}, allow_redirects=True, timeout=60)
    h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
    v = s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=60)
    if v.status_code == 200: break
    time.sleep(3)

# 拉第一条已完成任务
r = s.get(f"{BASE}/api/projects/{PROJECT}/tasks",
          params={"fields": "all", "page": 1, "page_size": 50}, headers=h, timeout=120)
r.raise_for_status()
tasks = r.json()
labeled = [t for t in tasks if t.get("is_labeled")]
print(f"first page: {len(tasks)} tasks, labeled: {len(labeled)}")
t = labeled[0] if labeled else tasks[0]

print("\n=== task keys ===")
print(sorted(t.keys()))
print("\n=== data keys ===")
print(sorted(t.get("data", {}).keys()))
img = t["data"].get("image", "")
print(f"\nimage field: starts_with={img[:40]!r}  len={len(img)}")
print(f"image looks like data-url base64: {img.startswith('data:image')}")
print(f"image looks like http URL: {img.startswith('http')}")

print("\n=== first annotation keys ===")
anns = t.get("annotations", [])
print(f"annotation count: {len(anns)}")
if anns:
    a = anns[0]
    print(f"annotation keys: {sorted(a.keys())}")
    res = a.get("result", [])
    print(f"  result[0] keys: {sorted(res[0].keys()) if res else 'none'}")
    if res:
        r0 = res[0]
        for k in ("from_name","to_name","type","original_width","original_height"):
            print(f"    {k}: {r0.get(k)}")
        print(f"    rle type={type(r0.get('rle')).__name__}  len={len(r0.get('rle',[]))}  first3={r0.get('rle',[])[:3]}")
        print(f"    brushlabels: {r0.get('brushlabels')}")
    print(f"  completed_by: {a.get('completed_by')}")

# 探针：拿这一条跑一次 save_brush_images_from_annotation，看产出文件名
os.makedirs(OUT_PROBE, exist_ok=True)
if anns:
    a = anns[0]
    from_name = a["result"][0]["from_name"]
    ls_brush.save_brush_images_from_annotation(
        task_id=t["id"], annotation_id=a["id"],
        completed_by=a.get("completed_by", ""),
        from_name=from_name, results=a["result"],
        out_dir=OUT_PROBE, out_format="png",
    )
    files = sorted(os.listdir(OUT_PROBE))
    print(f"\n=== probe out files ({len(files)}) ===")
    for f in files: print(" ", f)