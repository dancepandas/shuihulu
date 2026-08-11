"""把 task 882 的 prediction 转成 annotation, 测试是否渲染"""
import json, re, requests
BASE="http://localhost:8080"
s=requests.Session(); s.trust_env=False
g=s.get(f"{BASE}/user/login/",timeout=30)
m=re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"',g.text)
s.post(f"{BASE}/user/login/",data={"email":"chs9710@163.com","password":"Chengs1997","csrfmiddlewaretoken":m.group(1)},headers={"Referer":f"{BASE}/user/login/"},timeout=30)
h={"X-CSRFToken":s.cookies.get("csrftoken","")}

TID=882
t=s.get(f"{BASE}/api/projects/7/tasks/?fields=predictions&per_page=50",headers=h,timeout=30).json()
task=next(x for x in t if x["id"]==TID)
pred=task["predictions"][0]
result=pred["result"]
print(f"task {TID}: prediction has {len(result)} brush regions, model_version={pred.get('model_version')}")

# POST 一个 annotation
body={"result":result,"was_cancelled":False,"result_count":len(result)}
r=s.post(f"{BASE}/api/tasks/{TID}/annotations",json=body,headers={**h,"Content-Type":"application/json"},timeout=60)
print("create annotation:",r.status_code, r.text[:200])
