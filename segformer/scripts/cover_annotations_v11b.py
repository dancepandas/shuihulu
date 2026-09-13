"""用 SegFormer-v11b 的预测结果覆盖 LS project 10 的全部标注。
对每个任务: 删除已有 annotation → 将模型预测写为新 annotation (origin=manual)。

流程:
  1. 分页拉全部 679 个任务
  2. 下载原图 (复用缓存 data/ls_export_p10/images/)
  3. v11b 推理 5 类 mask
  4. 每类一个 brushlabels result (mask2rle)
  5. 删除该任务全部 annotations → POST 新 annotation (模型预测)

类别: 0=water 1=water_hyacinth 2=hard_structure 3=shore_vegetation 4=other_aquatic_vegetation
"""
import os, re, json, time
from collections import Counter
import cv2, numpy as np, torch, torch.nn.functional as F
import requests
from transformers import SegformerForSemanticSegmentation
from label_studio_sdk.converter.brush import mask2rle

ROOT = r"D:\chengs\9.project\shuihulu"
MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v11b")
LS_BASE = "http://192.168.30.107:8090"
LS_EMAIL, LS_PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
IMG_CACHE_DIR = os.path.join(ROOT, "data", "ls_export_p10", "images")
SZ = 640
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FROM_NAME, TO_NAME = "label", "image"
CLASS_IDS = [0, 1, 2, 3, 4]
LABEL_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
               3: "shore_vegetation", 4: "other_aquatic_vegetation"}


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


def fetch_all_tasks(s, h):
    tasks, page = [], 1
    while True:
        ok = False
        for retry in range(6):
            try:
                r = s.get(f"{LS_BASE}/api/projects/{PROJECT}/tasks",
                          params={"fields": "id,data", "page": page, "page_size": 100},
                          headers=h, timeout=300)
                r.raise_for_status(); batch = r.json(); ok = True; break
            except Exception as e:
                print(f"  page {page} retry {retry+1}: {type(e).__name__}")
                time.sleep(min(2**retry, 20))
        if not ok or not batch: break
        tasks.extend(batch)
        print(f"  page {page}: {len(batch)} 张 (cum {len(tasks)})")
        if len(batch) < 100: break
        page += 1
    return tasks


def download_image(s, h, rel_path):
    url = f"{LS_BASE}{rel_path}" if rel_path.startswith("/") else f"{LS_BASE}/data/{rel_path}"
    fname = os.path.basename(rel_path)
    local = os.path.join(IMG_CACHE_DIR, fname)
    if os.path.exists(local) and os.path.getsize(local) > 1024:
        return local
    for retry in range(6):
        try:
            r = s.get(url, headers=h, timeout=300)
            if r.status_code == 200 and len(r.content) > 1024:
                with open(local, "wb") as f:
                    f.write(r.content)
                return local
        except Exception:
            pass
        time.sleep(min(2**retry, 20))
    raise RuntimeError(f"download failed: {rel_path}")


def predict_mask(model, img_bgr, dev):
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (SZ, SZ), interpolation=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(((r - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        logits = F.interpolate(model(pixel_values=t).logits, size=(H, W),
                               mode="bilinear", align_corners=False)
    return logits[0].argmax(0).cpu().numpy()


def mask_to_results(class_map):
    H, W = class_map.shape[:2]
    results = []
    for c in CLASS_IDS:
        mask = (class_map == c).astype(np.uint8) * 255
        if not mask.any():
            continue
        results.append({
            "from_name": FROM_NAME, "to_name": TO_NAME,
            "type": "brushlabels",
            "value": {"format": "rle", "rle": mask2rle(mask),
                      "brushlabels": [LABEL_NAMES[c]]},
            "original_width": W, "original_height": H,
            "image_rotation": 0, "origin": "manual",
        })
    return results


def delete_annotations(s, h, tid):
    """删除某任务全部 annotations (只取 id, 不拉完整 result)"""
    r = s.get(f"{LS_BASE}/api/tasks/{tid}/annotations",
              params={"fields": "id"}, headers=h, timeout=120)
    anns = r.json() if r.status_code == 200 else []
    for a in anns:
        s.delete(f"{LS_BASE}/api/annotations/{a['id']}", headers=h, timeout=120)


def post_annotation(s, h, tid, results):
    body = {"result": results, "ground_truth": True}
    r = s.post(f"{LS_BASE}/api/tasks/{tid}/annotations", json=body,
               headers={**h, "Content-Type": "application/json"}, timeout=180)
    return r.status_code


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] 加载 v11b: {MODEL_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).eval().to(dev)
    print(f"  模型就绪, dev={dev}")

    s, h = ls_login()
    print("[2/3] 拉取全部任务...")
    tasks = fetch_all_tasks(s, h)
    print(f"  任务总数: {len(tasks)}")

    print("[3/3] 推理 + 覆盖 annotation...")
    ok = 0; skip = 0; cnt = Counter()
    for i, t in enumerate(tasks):
        tid = t["id"]
        rel = t["data"]["image"]
        try:
            img_path = download_image(s, h, rel)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [skip] task {tid}: decode fail"); skip += 1; continue
            class_map = predict_mask(model, img, dev)
            results = mask_to_results(class_map)
            if not results:
                print(f"  [skip] task {tid}: 无前景类"); skip += 1; continue
            # 删除旧 annotation
            delete_annotations(s, h, tid)
            # POST 新 annotation (模型预测覆盖)
            status = post_annotation(s, h, tid, results)
            if status in (200, 201):
                ok += 1
                for r_ in results:
                    cnt[r_["value"]["brushlabels"][0]] += 1
            else:
                print(f"  [fail] task {tid}: POST HTTP {status}")
                skip += 1
        except Exception as e:
            print(f"  [skip] task {tid}: {type(e).__name__} {e}")
            skip += 1
        if (i + 1) % 20 == 0 or (i + 1) == len(tasks):
            print(f"  [{i+1}/{len(tasks)}] ok={ok} skip={skip}")

    print(f"\nDONE: {ok}/{len(tasks)} 覆盖成功, skip={skip}")
    print("类别覆盖统计:")
    for name, n in cnt.most_common():
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
