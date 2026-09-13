"""拉取 LS project 10 全部 completed 标注 → 训练数据集 v12 (补充"几乎全屏水葫芦"难例)。
与 pull_all_p10.py 不同: 支持 base64 内嵌 image (新导入的截图) 与相对路径两种格式。

流程: 拉全部 is_labeled 任务 → 取最新 annotation → 解码 image → 合并 5 类 mask → 切 train/val
类别: 0=water 1=water_hyacinth 2=hard_structure 3=shore_vegetation 4=other_aquatic_vegetation
"""
import os, re, json, time, shutil, random, base64, glob
from collections import Counter
import cv2, numpy as np, requests
from label_studio_sdk.converter.brush import decode_rle

ROOT = r"D:\chengs\9.project\shuihulu"
LS_BASE = "http://192.168.30.107:8090"
LS_EMAIL, LS_PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
IMG_CACHE_DIR = os.path.join(ROOT, "data", "ls_export_p10", "images")
OUT_DIR = os.path.join(ROOT, "datasets", "hyacinth_ls_v12")

LABEL2ID = {"water": 0, "water_hyacinth": 1, "hard_structure": 2,
            "shore_vegetation": 3, "other_aquatic_vegetation": 4}
ID2NAME = {v: k for k, v in LABEL2ID.items()}


def ls_login():
    s = requests.Session(); s.trust_env = False
    for attempt in range(10):
        try:
            g = s.get(f"{LS_BASE}/user/login/", timeout=120)
            m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
            s.post(f"{LS_BASE}/user/login/",
                   data={"email": LS_EMAIL, "password": LS_PWD, "csrfmiddlewaretoken": m.group(1)},
                   headers={"Referer": f"{LS_BASE}/user/login/"}, allow_redirects=True, timeout=120)
            h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
            if s.get(f"{LS_BASE}/api/projects/{PROJECT}", headers=h, timeout=120).status_code == 200:
                print(f"[login] OK (attempt {attempt+1})")
                return s, h
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("LS login failed")


def fetch_completed(s, h):
    tasks, page = [], 1
    while True:
        ok = False
        for retry in range(6):
            try:
                r = s.get(f"{LS_BASE}/api/projects/{PROJECT}/tasks",
                          params={"fields": "all", "page": page, "page_size": 100},
                          headers=h, timeout=300)
                r.raise_for_status(); batch = r.json(); ok = True; break
            except Exception as e:
                print(f"  page {page} retry {retry+1}: {type(e).__name__}")
                time.sleep(min(2**retry, 20))
        if not ok or not batch: break
        tasks.extend(t for t in batch if t.get("is_labeled"))
        if len(batch) < 100: break
        page += 1
    return tasks


def latest_annotation(t):
    anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    anns.sort(key=lambda a: a.get("created_at", ""))
    return anns[-1]


def load_image(s, h, data_image):
    """支持 base64 内嵌 与 相对路径两种 image 格式, 返回 BGR numpy"""
    if data_image.startswith("data:image"):
        # base64 内嵌
        _, encoded = data_image.split(",", 1)
        buf = base64.b64decode(encoded)
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return img
    # 相对路径
    url = f"{LS_BASE}{data_image}" if data_image.startswith("/") else f"{LS_BASE}/data/{data_image}"
    fname = os.path.basename(data_image)
    local = os.path.join(IMG_CACHE_DIR, fname)
    if not (os.path.exists(local) and os.path.getsize(local) > 1024):
        for retry in range(6):
            try:
                r = s.get(url, headers=h, timeout=300)
                if r.status_code == 200 and len(r.content) > 1024:
                    with open(local, "wb") as f:
                        f.write(r.content)
                    break
            except Exception:
                pass
            time.sleep(min(2**retry, 20))
    img = cv2.imread(local, cv2.IMREAD_COLOR)
    return img


def decode_mask(rle, H, W):
    img = decode_rle(rle)
    if img.size == H * W * 4:
        return np.reshape(img, [H, W, 4])[:, :, 3]
    return np.reshape(img, [H, W])


def build_mask(annotation, H, W):
    merged = np.zeros((H, W), dtype=np.uint8)
    for res in annotation.get("result", []):
        if (res.get("type") or "").lower() != "brushlabels":
            continue
        val = res.get("value") or {}
        rle = val.get("rle")
        labels = val.get("brushlabels") or []
        if not rle or not labels or labels[0] not in LABEL2ID:
            continue
        cls_id = LABEL2ID[labels[0]]
        alpha = decode_mask(rle, H, W)
        merged[alpha > 0] = cls_id
    return merged


def main():
    os.makedirs(IMG_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)

    s, h = ls_login()
    print("[1/3] 拉取 completed 任务...")
    tasks = fetch_completed(s, h)
    print(f"  completed 任务: {len(tasks)}")

    print("[2/3] 下载/解码 image + 合并 mask...")
    done = skip = 0
    base64_cnt = 0
    for i, t in enumerate(tasks):
        tid = t["id"]
        ann = latest_annotation(t)
        if ann is None:
            skip += 1; continue
        try:
            img = load_image(s, h, t["data"]["image"])
            if img is None:
                print(f"  [skip] task {tid}: image decode fail"); skip += 1; continue
            if t["data"]["image"].startswith("data:image"):
                base64_cnt += 1
            H, W = img.shape[:2]
            mask = build_mask(ann, H, W)
        except Exception as e:
            print(f"  [skip] task {tid}: {type(e).__name__}"); skip += 1; continue

        stem = f"p10_{tid:05d}"
        cv2.imwrite(os.path.join(OUT_DIR, "images", f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(OUT_DIR, "masks", f"{stem}.png"), mask)
        done += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
            print(f"  [{i+1}/{len(tasks)}] done={done} skip={skip}")

    # 类别分布
    print(f"\nbase64 内嵌图: {base64_cnt} 张")
    counts = np.zeros(5, dtype=np.int64); total = 0
    for f in glob.glob(os.path.join(OUT_DIR, "masks", "*.png")):
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        total += m.size
        for c in range(5):
            counts[c] += int((m == c).sum())
    print("类别像素分布:")
    for c in range(5):
        print(f"  {c} {ID2NAME[c]:28s} {counts[c]:>12,} px  {counts[c]/total*100:5.2f}%")

    # 切 train/val
    print("\n[3/3] 切 train/val 8:2...")
    random.seed(42)
    imgs = sorted(glob.glob(os.path.join(OUT_DIR, "images", "*.jpg")))
    random.shuffle(imgs)
    n_train = int(len(imgs) * 0.8)
    for split, lst in [("train", imgs[:n_train]), ("val", imgs[n_train:])]:
        id_ = os.path.join(OUT_DIR, "images", split); md_ = os.path.join(OUT_DIR, "masks", split)
        os.makedirs(id_, exist_ok=True); os.makedirs(md_, exist_ok=True)
        for p in lst:
            stem = os.path.splitext(os.path.basename(p))[0]
            shutil.move(p, os.path.join(id_, os.path.basename(p)))
            shutil.move(os.path.join(OUT_DIR, "masks", f"{stem}.png"),
                        os.path.join(md_, f"{stem}.png"))
    print(f"  train: {n_train}, val: {len(imgs)-n_train}")
    print(f"\nDONE. 数据集: {OUT_DIR}")


if __name__ == "__main__":
    main()
