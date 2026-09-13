"""拉取 LS project 10 里 7 张 base64 内嵌图 (data/全面积截图, 已人工调整) 的标注,
追加到 v11 训练集 (datasets/hyacinth_ls_v11/images/train + masks/train)。
"""
import os, re, json, time, base64, glob
from collections import Counter
import cv2, numpy as np, requests
from label_studio_sdk.converter.brush import decode_rle

ROOT = r"D:\chengs\9.project\shuihulu"
LS_BASE = "http://192.168.30.107:8090"
LS_EMAIL, LS_PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
OUT_TRAIN_IMG = os.path.join(ROOT, "datasets", "hyacinth_ls_v11", "images", "train")
OUT_TRAIN_MSK = os.path.join(ROOT, "datasets", "hyacinth_ls_v11", "masks", "train")

LABEL2ID = {"water": 0, "water_hyacinth": 1, "hard_structure": 2,
            "shore_vegetation": 3, "other_aquatic_vegetation": 4}


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
                return s, h
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("login failed")


def latest_annotation(t):
    anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
    if not anns:
        return None
    anns.sort(key=lambda a: a.get("created_at", ""))
    return anns[-1]


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
        rle = val.get("rle"); labels = val.get("brushlabels") or []
        if not rle or not labels or labels[0] not in LABEL2ID:
            continue
        alpha = decode_mask(rle, H, W)
        merged[alpha > 0] = LABEL2ID[labels[0]]
    return merged


def main():
    os.makedirs(OUT_TRAIN_IMG, exist_ok=True)
    os.makedirs(OUT_TRAIN_MSK, exist_ok=True)
    s, h = ls_login()

    # 拉全部任务, 筛 base64 image (即 7 张截图)
    tasks = []
    page = 1
    while True:
        r = s.get(f"{LS_BASE}/api/projects/{PROJECT}/tasks",
                  params={"fields": "all", "page": page, "page_size": 100},
                  headers=h, timeout=300)
        batch = r.json()
        if not batch: break
        tasks.extend(t for t in batch if t["data"]["image"].startswith("data:image"))
        if len(batch) < 100: break
        page += 1
    print(f"base64 内嵌图: {len(tasks)} 张")

    done = 0
    for t in tasks:
        tid = t["id"]
        ann = latest_annotation(t)
        if ann is None:
            print(f"  [skip] task {tid}: 无 annotation"); continue
        encoded = t["data"]["image"].split(",", 1)[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] task {tid}: decode fail"); continue
        H, W = img.shape[:2]
        mask = build_mask(ann, H, W)
        stem = f"p10_{tid:05d}"
        cv2.imwrite(os.path.join(OUT_TRAIN_IMG, f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(OUT_TRAIN_MSK, f"{stem}.png"), mask)
        cnt = Counter(mask.ravel())
        print(f"  task {tid} ({W}x{H}): " + ", ".join(
            f"{LABEL2ID and [k for k,v in LABEL2ID.items() if v==c][0]}={cnt.get(c,0)}px"
            for c in range(5) if cnt.get(c,0) > 0))
        done += 1

    print(f"\nDONE: {done}/{len(tasks)} 张已追加到 v11 train 集")
    train_n = len(glob.glob(os.path.join(OUT_TRAIN_IMG, "*.jpg")))
    val_n = len(glob.glob(os.path.join(ROOT, "datasets", "hyacinth_ls_v11", "images", "val", "*.jpg")))
    print(f"  v11 train 现在: {train_n} 张, val: {val_n} 张")


if __name__ == "__main__":
    main()
