"""用 v11b 标注 data/全面积 的截图并导入 LS project 10 (带 annotation)。
流程: 读图 → v11b 推理 5 类 mask → annotation result → import 导入。
"""
import os, re, json, time, base64, glob
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
SRC_DIR = os.path.join(ROOT, "data", "全面积")
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


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] 加载 v11b: {MODEL_PATH}")
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).eval().to(dev)
    print(f"  模型就绪, dev={dev}")

    files = sorted(glob.glob(os.path.join(SRC_DIR, "DJI_*.jpeg")))
    print(f"[2/3] 找到 {len(files)} 张 DJI 影像")

    s, h = ls_login()

    print("[3/3] 推理 + 导入...")
    tasks = []
    cnt = Counter()
    for f in files:
        img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] {os.path.basename(f)}: decode fail"); continue
        class_map = predict_mask(model, img, dev)
        results = mask_to_results(class_map)
        if not results:
            print(f"  [skip] {os.path.basename(f)}: 无前景类"); continue
        # 转 JPEG base64 (减小体积)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        tasks.append({"data": {"image": b64},
                      "annotations": [{"result": results, "ground_truth": True}]})
        for r_ in results:
            cnt[r_["value"]["brushlabels"][0]] += 1
        print(f"  {os.path.basename(f)}: {len(results)} 类 "
              f"{[r_['value']['brushlabels'][0] for r_ in results]}")

    # 导入
    imported = 0
    for i in range(0, len(tasks), 3):
        batch = tasks[i:i+3]
        for retry in range(6):
            try:
                r = s.post(f"{LS_BASE}/api/projects/{PROJECT}/import", json=batch,
                           headers={**h, "Content-Type": "application/json"}, timeout=180)
                if r.status_code in (200, 201):
                    imported += len(batch); break
                print(f"  batch {i//3+1}: HTTP {r.status_code}, retry {retry+1}")
            except Exception as e:
                print(f"  batch {i//3+1}: {type(e).__name__}, retry {retry+1}")
            time.sleep(min(2**retry, 20))

    print(f"\nDONE: 导入 {imported}/{len(tasks)} 张")
    print("类别统计:")
    for name, n in cnt.most_common():
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
