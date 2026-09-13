"""测试健壮登录闭环：fresh login -> 立刻验证 API，直到拿到稳定会话。"""
import requests, re, time

base = 'http://192.168.30.107:8090'
EMAIL, PWD = 'chs9710@163.com', 'cHENGS1997'


def fresh_login_attempt():
    s = requests.Session()
    s.trust_env = False
    g = s.get(f'{base}/user/login/', timeout=60)
    if g.status_code != 200:
        return None, f'login page HTTP {g.status_code}'
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text)
    csrf = m.group(1) if m else None
    if not csrf:
        return None, 'no csrf'
    p = s.post(f'{base}/user/login/',
               data={'email': EMAIL, 'password': PWD, 'csrfmiddlewaretoken': csrf},
               headers={'Referer': f'{base}/user/login/'}, allow_redirects=True, timeout=60)
    has_sid = 'sessionid' in s.cookies
    h = {'X-CSRFToken': s.cookies.get('csrftoken', '')}
    # 立刻验证
    v = s.get(f'{base}/api/projects/3/tasks?page_size=1', headers=h, timeout=60)
    if v.status_code == 200:
        return (s, h), f'OK (sessionid={has_sid})'
    return None, f'login post={p.status_code} sid={has_sid} verify={v.status_code} body={v.text[:80]}'


print('waiting 5s (避免被限流)...')
time.sleep(5)
for i in range(8):
    res, msg = fresh_login_attempt()
    print(f'attempt {i+1}: {msg}')
    if res:
        s, h = res
        # 顺势验证 project 4 也能访问
        v4 = s.get(f'{base}/api/projects/4/tasks?page_size=1', headers=h, timeout=60)
        print(f'  project 4 verify: {v4.status_code}')
        if v4.status_code == 200:
            print('  -> STABLE SESSION ACQUIRED')
            break
    time.sleep(3)
else:
    print('FAILED to get stable session')
