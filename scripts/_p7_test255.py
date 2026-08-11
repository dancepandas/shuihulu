"""task 882: 解码 prediction mask -> x255 -> 重编码 -> 更新 annotation, 测试 255 是否渲染"""
import re, requests
import numpy as np
from label_studio_sdk.converter.brush import decode_rle, mask2rle
BASE="http://localhost:8080"
s=requests.Session(); s.trust_env=False
g=s.get(f"{BASE}/user/login/",timeout=30)
m=re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"',g.text)
s.post(f"{BASE}/user/login/",data={"email":"chs9710@163.com","password":"Chengs1997","csrfmiddlewaretoken":m.group(1)},headers={"Referer":f"{BASE}/user/login/"},timeout=30)
h={"X-CSRFToken":s.cookies.get("csrftoken","")}

TID=882
t=s.get(f"{BASE}/api/projects/7/tasks/?fields=predictions&per_page=50",headers=h,timeout=30).json()
task=next(x for x in t if x["id"]==TID)
pred_result=task["predictions"][0]["result"]
print(f"task {TID}: {len(pred_result)} regions")

# 删旧 annotation (id=221)
s.delete(f"{BASE}/api/annotations/221",headers=h,timeout=30)

new_result=[]
for item in pred_result:
    W,H=item["original_width"],item["original_height"]
    dec=decode_rle(item["value"]["rle"])
    alpha=np.reshape(dec,[H,W,4])[:,:,3]          # 0/1 mask
    mask=(alpha*255).astype(np.uint8)             # 0/255
    rle=mask2rle(mask)
    new_result.append({
        "original_width":W,"original_height":H,"image_rotation":0,
        "value":{"format":"rle","rle":rle,"brushlabels":item["value"]["brushlabels"]},
        "id":item.get("id") or "x"+np.base_repr(np.random.choice(99999),36).lower(),
        "from_name":"label","to_name":"image","type":"brushlabels","origin":"manual",
    })
print("new fg value check: decode one region")
d=decode_rle(new_result[0]["value"]["rle"]); a=np.reshape(d,[new_result[0]["original_height"],new_result[0]["original_width"],4])[:,:,3]
print("  fg unique:",np.unique(a[a>0]))

r=s.post(f"{BASE}/api/tasks/{TID}/annotations",json={"result":new_result,"was_cancelled":False,"result_count":len(new_result)},
         headers={**h,"Content-Type":"application/json"},timeout=60)
print("create annotation:",r.status_code)
