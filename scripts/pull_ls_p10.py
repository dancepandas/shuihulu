"""
拉取 Label Studio project 10 已完成任务的：
  - 原图（image 字段是相对路径 /data/upload/10/<hash>_<name>.jpeg，拼接 BASE 下载）
  - 每条笔刷标注一个 PNG，按 LS brush converter 命名规则
    task-{tid}-annotation-{aid}-by-{completed_by}-{sanitized_label}-{i}.png

输出：
  data/ls_export_p10/
    images/...jpeg              原始图
    task-...png                 每标签一张 PNG（已分 channel=alpha 的 0/255 二值）
  data/ls_export_p10_meta.jsonl 每条任务一条完整 annotation JSON（含 value.rle）

新标签体系（5 类，项目 10 自带）：
  water / shore_vegetation / water_hyacinth / hard_structure / other_aquatic_vegetation
"""
import os, re, json, time, requests
from collections import defaultdict
from label_studio_sdk.converter import brush as ls_brush

BASE = "http://192.168.30.107:8090"
EMAIL, PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
ROOT = r"D:\chengs\9.project\shuihulu"
OUT_DIR = os.path.join(ROOT, "data", "ls_export_p10")
IMG_DIR = os.path.join(OUT_DIR, "images")
META_PATH = os.path.join(OUT_DIR, "meta.jsonl")
LOG_PATH = os.path.join(OUT_DIR, "download.log")
PROGRESS_EVERY = 25

# ---------- 登录 ----------
s = requests.Session()
s.trust_env = False
for i in range(10):
    g = s.get(f"{BASE}/user/login/", timeout=60)
    if g.status_code != 200:
        time.sleep(3); continue
    csrf_m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
    if not csrf_m:
        time.sleep(3); continue
    s.post(f"{BASE}/user/login/",
           data={"email": EMAIL, "password": PWD, "csrfmiddlewaretoken": csrf_m.group(1)},
           headers={"Referer": f"{BASE}/user/login/"}, allow_redirects=True, timeout=60)
    h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
    v = s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=60)
    if v.status_code == 200:
        print(f"[login] OK (attempt {i+1})")
        break
    time.sleep(3)
else:
    raise SystemExit("login failed")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
log = open(LOG_PATH, "a", encoding="utf-8")
meta_f = open(META_PATH, "a", encoding="utf-8")

def logln(msg):
    print(msg); log.write(msg + "\n"); log.flush()

# ---------- 分页拉全部 ----------
logln(f"[fetch] project {PROJECT} tasks (labeled only) ...")
all_tasks = []
page = 1
while True:
    ok = False
    for retry in range(6):
        try:
            r = s.get(f"{BASE}/api/projects/{PROJECT}/tasks",
                      params={"fields": "all", "page": page, "page_size": 100},
                      headers=h, timeout=300)
            r.raise_for_status()
            batch = r.json()
            ok = True; break
        except Exception as e:
            logln(f"  page {page} retry {retry+1}: {type(e).__name__}")
            time.sleep(min(2 ** retry, 20))
    if not ok:
        logln(f"  page {page}: FAILED, stop")
        break
    if not batch: break
    all_tasks.extend(batch)
    logln(f"  page {page}: {len(batch)} (cum {len(all_tasks)})")
    if len(batch) < 100: break
    page += 1
logln(f"[fetch] total {len(all_tasks)} tasks")

labeled = [t for t in all_tasks if t.get("is_labeled")]
logln(f"[fetch] is_labeled = {len(labeled)}")

# ---------- 缓存已下载图 ----------
img_cache = {}
def fetch_image(rel_path):
    """rel_path = /data/upload/10/<hash>_<name>.jpeg → bytes"""
    if rel_path in img_cache: return img_cache[rel_path]
    url = f"{BASE}{rel_path}" if rel_path.startswith("/") else f"{BASE}/data/{rel_path}"
    for retry in range(6):
        try:
            r = s.get(url, headers=h, timeout=300)
            if r.status_code == 200 and len(r.content) > 1024:
                img_cache[rel_path] = r.content
                return r.content
            logln(f"  img retry {retry+1}: HTTP {r.status_code} len={len(r.content)}")
        except Exception as e:
            logln(f"  img retry {retry+1}: {type(e).__name__}")
        time.sleep(min(2 ** retry, 20))
    raise RuntimeError(f"image download failed: {rel_path}")

# ---------- 主循环：保存原图 + 笔刷 PNG ----------
ok_img = ok_png = 0
skip = 0
for idx, t in enumerate(labeled):
    tid = t["id"]
    anns = t.get("annotations", [])
    if not anns:
        skip += 1; continue
    rel = t["data"]["image"]
    fname = os.path.basename(rel)
    img_path = os.path.join(IMG_DIR, fname)
    if not os.path.exists(img_path):
        try:
            data = fetch_image(rel)
            with open(img_path, "wb") as f: f.write(data)
            ok_img += 1
        except Exception as e:
            logln(f"  [skip] task {tid}: img {e}")
            skip += 1; continue
    else:
        ok_img += 1   # 已下载过就计数

    # 每条 annotation → 每 result 一张 PNG
    for a in anns:
        results = a.get("result", [])
        if not results: continue
        from_name = results[0].get("from_name", "label")
        try:
            ls_brush.save_brush_images_from_annotation(
                task_id=tid, annotation_id=a["id"],
                completed_by=a.get("completed_by", ""),
                from_name=from_name, results=results,
                out_dir=OUT_DIR, out_format="png",
            )
            ok_png += sum(1 for r in results if r.get("type") == "brushlabels")
        except Exception as e:
            logln(f"  [warn] task {tid} ann {a['id']}: brush decode {e}")

    # 元数据
    meta_f.write(json.dumps(t, ensure_ascii=False) + "\n")

    if (idx + 1) % PROGRESS_EVERY == 0 or (idx + 1) == len(labeled):
        logln(f"  [{idx+1}/{len(labeled)}] imgs={ok_img}  pngs={ok_png}  skip={skip}")

meta_f.close(); log.close()
logln(f"\nDONE  imgs={ok_img}  pngs={ok_png}  skip={skip}")
logln(f"      images: {IMG_DIR}")
logln(f"      meta:   {META_PATH}")