"""仅导入 v9_500.jsonl (不清空，直接追加)"""
import json, time, re
import requests

BASE = "http://192.168.30.107:8090"
JSONL = "data/v9_500.jsonl"
PROJECT = 3

# --- 登录 ---
s = requests.Session()
s.trust_env = False
for attempt in range(1, 30):
    try:
        g = s.get(f"{BASE}/user/login/", timeout=60)
        m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
        csrf = m.group(1) if m else None
        s.post(f"{BASE}/user/login/",
               data={"email": "chs9710@163.com", "password": "cHENGS1997", "csrfmiddlewaretoken": csrf},
               headers={"Referer": f"{BASE}/user/login/"}, allow_redirects=True, timeout=60)
        h = {"X-CSRFToken": s.cookies.get("csrftoken", "")}
        v = s.get(f"{BASE}/api/projects/{PROJECT}", headers=h, timeout=120)
        if v.status_code == 200:
            print(f"login OK (attempt {attempt})")
            break
        else:
            print(f"  verify attempt {attempt}: HTTP {v.status_code}")
    except Exception as e:
        print(f"  login attempt {attempt}: {e}")
    time.sleep(5)
else:
    raise RuntimeError("login failed")

with open(JSONL, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]
print(f"loaded {len(records)} records")

imported = 0
for i in range(0, len(records), 10):  # 10 per batch, easier on server
    batch = records[i:i + 10]
    success = False
    for retry in range(8):
        try:
            r = s.post(f"{BASE}/api/projects/{PROJECT}/import",
                       json=batch, headers={**h, "Content-Type": "application/json"}, timeout=300)
            if r.status_code in (200, 201):
                imported += len(batch)
                success = True
                break
            print(f"  batch {i//10+1}/{len(records)//10+1}: HTTP {r.status_code}, retry {retry+1}")
        except Exception as e:
            print(f"  batch {i//10+1}: {type(e).__name__}, retry {retry+1}")
        time.sleep(min(2 ** retry, 20))
    if not success:
        print(f"  batch {i//10+1}: FAILED after retries")
    if (i // 10 + 1) % 10 == 0:
        print(f"  === {imported}/{len(records)} ===")

print(f"\nDONE: {imported}/{len(records)}")
