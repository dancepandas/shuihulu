"""dry-run: 单张预标注推送 + 回读验证"""
import os, re, sys, json, time
import cv2, numpy as np, torch, torch.nn.functional as F
import requests
from transformers import SegformerForSemanticSegmentation
from label_studio_sdk.converter.brush import mask2rle
sys.path.insert(0, r"D:\chengs\9.project\shuihulu\segformer\scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("pre", r"D:\chengs\9.project\shuihulu\segformer\scripts\preanno_v10_ls.py")
pre = importlib.util.module_from_spec(spec); spec.loader.exec_module(pre)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = SegformerForSemanticSegmentation.from_pretrained(pre.MODEL_PATH).eval().to(dev)
s, h = pre.ls_login()
unlabeled = pre.fetch_unlabeled(s, h)
t = unlabeled[0]
print(f"dry-run task: {t['id']}")

content = pre.fetch_image(s, h, t["data"]["image"], {})
img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
class_map = pre.predict_mask(model, img, dev)
results = pre.mask_to_results(class_map)
print(f"results: {len(results)} 类")
for r_ in results:
    lbl = r_["value"]["brushlabels"][0]
    mask = (class_map == list(pre.LABEL_NAMES.keys())[list(pre.LABEL_NAMES.values()).index(lbl)]).sum()
    print(f"  {lbl}: rle_len={len(r_['value']['rle'])}  px={mask}  keys={sorted(r_.keys())}  value_keys={sorted(r_['value'].keys())}")

# POST
pred = {"task": t["id"], "model_version": pre.MODEL_VERSION, "result": results}
r = s.post(f"{pre.LS_BASE}/api/predictions", json=pred,
           headers={**h, "Content-Type": "application/json"}, timeout=120)
print(f"POST -> HTTP {r.status_code}")
print(r.text[:300] if r.status_code >= 400 else "OK")

# 回读验证
if r.status_code in (200, 201):
    time.sleep(1)
    rr = s.get(f"{pre.LS_BASE}/api/predictions?project={pre.PROJECT}&page_size=5", headers=h, timeout=60)
    j = rr.json()
    print(f"\n回读 {len(j)} 条 predictions:")
    if j:
        p = j[0]
        print("  prediction keys:", sorted(p.keys()))
        res = p["result"][0]
        print("  result[0] keys:", sorted(res.keys()))
        print("  value keys:", sorted(res.get("value", {}).keys()))
        print("  brushlabels:", res.get("value", {}).get("brushlabels"))
        print("  rle len:", len(res.get("value", {}).get("rle", [])))
        print("  original_w/h:", res.get("original_width"), res.get("original_height"))
    # 清理这条测试
    # predictions 删除用 id
    if j:
        pid = j[0].get("id")
        dd = s.delete(f"{pre.LS_BASE}/api/predictions/{pid}", headers=h, timeout=60)
        print(f"  cleanup delete {pid} -> {dd.status_code}")
