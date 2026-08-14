"""用 SegFormer-v10 对 LS project 10 未标注任务做笔刷预标注。
流程:
  1. 分页拉 project 10 全部任务, 筛 is_labeled=false (未标注)
  2. 下载原图 (中文路径/相对路径处理)
  3. SegFormer-v10 推理 5 类 mask
  4. 每类一个 brushlabels result (mask2rle 位打包, top-level 尺寸, alpha=255)
  5. POST /api/predictions 推送 (裸格式 {"task": id, "model_version", "result"})
"""
import os, re, sys, json, time
from collections import Counter
import cv2, numpy as np, torch, torch.nn.functional as F
import requests
from transformers import SegformerForSemanticSegmentation
from label_studio_sdk.converter.brush import mask2rle

ROOT = r"D:\chengs\9.project\shuihulu"
MODEL_PATH = os.path.join(ROOT, "runs", "segformer", "segformer_b2_ls_v10")
LS_BASE = "http://192.168.30.107:8090"
LS_EMAIL, LS_PWD = "chs9710@163.com", "cHENGS1997"
PROJECT = 10
SZ = 640
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
# BrushLabels from_name/to_name (来自项目 labeling_config)
FROM_NAME, TO_NAME = "label", "image"
# v10 5 类: 全部推送 (含 water 背景类, 标注员有完整预标注)
CLASS_IDS = [0, 1, 2, 3, 4]
LABEL_NAMES = {0: "water", 1: "water_hyacinth", 2: "hard_structure",
               3: "shore_vegetation", 4: "other_aquatic_vegetation"}
MODEL_VERSION = "segformer-b2-v10-640"

# ---- LS 登录 ----
def ls_login():
    s = requests.Session(); s.trust_env = False
    for attempt in range(10):
        try:
            g = s.get(f"{LS_BASE}/user/login/", timeout=120)
            m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
            csrf = m.group(1) if m else None
            s.post(f"{LS_BASE}/user/login/",
                   data={"email": LS_EMAIL, "password": LS_PWD, "csrfmiddlewaretoken": csrf},
                   headers={"Referer": f"{LS_BASE}/user/login/"}, allow_redirects=True, timeout=120)
            h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
            v = s.get(f"{LS_BASE}/api/projects/{PROJECT}", headers=h, timeout=120)
            if v.status_code == 200:
                print(f"[login] OK (attempt {attempt+1})")
                return s, h
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("LS login failed")

# ---- 拉未标注任务 ----
def fetch_unlabeled(s, h):
    tasks = []
    page = 1
    while True:
        ok = False
        for retry in range(6):
            try:
                r = s.get(f"{LS_BASE}/api/projects/{PROJECT}/tasks",
                          params={"fields": "all", "page": page, "page_size": 100},
                          headers=h, timeout=300)
                r.raise_for_status()
                batch = r.json(); ok = True; break
            except Exception as e:
                print(f"  page {page} retry {retry+1}: {type(e).__name__}")
                time.sleep(min(2**retry, 20))
        if not ok or not batch: break
        tasks.extend(t for t in batch if not t.get("is_labeled"))
        if len(batch) < 100: break
        page += 1
    return tasks

# ---- 下载图片 (相对路径 → 完整 URL) ----
def fetch_image(s, h, rel_path, cache):
    if rel_path in cache: return cache[rel_path]
    url = f"{LS_BASE}{rel_path}" if rel_path.startswith("/") else f"{LS_BASE}/data/{rel_path}"
    for retry in range(6):
        try:
            r = s.get(url, headers=h, timeout=300)
            if r.status_code == 200 and len(r.content) > 1024:
                cache[rel_path] = r.content
                return r.content
        except Exception:
            pass
        time.sleep(min(2**retry, 20))
    raise RuntimeError(f"image download failed: {rel_path}")

# ---- v10 推理 → 类别 mask ----
def predict_mask(model, img_bgr, dev):
    H, W = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (SZ, SZ), interpolation=cv2.INTER_LINEAR) / 255.0
    t = torch.from_numpy(((r - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).float().to(dev)
    with torch.no_grad():
        logits = F.interpolate(model(pixel_values=t).logits, size=(H, W),
                               mode="bilinear", align_corners=False)
    return logits[0].argmax(0).cpu().numpy()

# ---- mask → LS brush prediction result ----
def mask_to_results(class_map):
    H, W = class_map.shape[:2]
    results = []
    for c in CLASS_IDS:
        mask = (class_map == c).astype(np.uint8) * 255
        if not mask.any(): continue   # 无该类别跳过
        results.append({
            "from_name": FROM_NAME, "to_name": TO_NAME,
            "type": "brushlabels",
            "value": {"format": "rle", "rle": mask2rle(mask),
                      "brushlabels": [LABEL_NAMES[c]]},
            "original_width": W, "original_height": H,
            "image_rotation": 0,
        })
    return results

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/4] 加载 SegFormer-v10: {MODEL_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).eval().to(dev)
    print(f"  模型就绪, dev={dev}")

    s, h = ls_login()
    print("[2/4] 拉取未标注任务...")
    unlabeled = fetch_unlabeled(s, h)
    print(f"  未标注任务: {len(unlabeled)}")

    print(f"[3/4] 推理 + 生成预标注...")
    img_cache = {}
    preds_payload = []
    n_with_pred = 0
    cnt = Counter()
    for i, t in enumerate(unlabeled):
        tid = t["id"]
        rel = t["data"]["image"]
        try:
            content = fetch_image(s, h, rel, img_cache)
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [skip] task {tid}: decode fail"); continue
            class_map = predict_mask(model, img, dev)
            results = mask_to_results(class_map)
            if results:
                preds_payload.append({"task": tid, "model_version": MODEL_VERSION,
                                      "result": results})
                n_with_pred += 1
                for r_ in results:
                    cnt[r_["value"]["brushlabels"][0]] += 1
        except Exception as e:
            print(f"  [skip] task {tid}: {e}")
        if (i + 1) % 20 == 0 or (i + 1) == len(unlabeled):
            print(f"  [{i+1}/{len(unlabeled)}] 有预标注 {n_with_pred}")

    print(f"  生成 {len(preds_payload)} 条预标注 (有类别 {n_with_pred} 张)")
    for name, n in cnt.most_common():
        print(f"    {name}: {n}")

    print(f"[4/4] 推送 predictions 到 project {PROJECT}...")
    # 逐条 POST (LS 批量接口不可靠, 逐条稳妥)
    ok = 0
    for i, pred in enumerate(preds_payload):
        for retry in range(5):
            try:
                r = s.post(f"{LS_BASE}/api/predictions", json=pred,
                           headers={**h, "Content-Type": "application/json"}, timeout=120)
                if r.status_code in (200, 201):
                    ok += 1; break
                print(f"  [{i+1}] task {pred['task']} HTTP {r.status_code}: {r.text[:100]}")
                time.sleep(2)
            except Exception as e:
                print(f"  [{i+1}] task {pred['task']} exc {e}")
                time.sleep(2)
        if (i + 1) % 20 == 0:
            print(f"  {ok}/{i+1} 推送完成")
    print(f"\nDONE: {ok}/{len(preds_payload)} predictions 推送成功")

if __name__ == "__main__":
    main()
