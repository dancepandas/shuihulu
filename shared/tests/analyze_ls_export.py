import json

with open("D:/chengs/9.project/shuihulu/datasets/project-3-at-2026-05-14-10-37-b3c9ac5a.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total items: {len(data)}")

for i, item in enumerate(data[:3]):
    anns = item.get("annotations", [])
    if anns:
        results = anns[0].get("result", [])
        print(f"  Item {i}: {len(results)} annotations")
        if results:
            r = results[0]
            print(f"    type: {r.get('type')}")
            print(f"    value keys: {list(r['value'].keys())}")
            print(f"    from_name: {r.get('from_name')}, to_name: {r.get('to_name')}")
            print(f"    original_width: {r.get('original_width')}, original_height: {r.get('original_height')}")
            if "points" in r["value"]:
                print(f"    points count: {len(r['value']['points'])}")
                print(f"    first point: {r['value']['points'][0]}")
            if "rectanglelabels" in r["value"]:
                print(f"    rectanglelabels: {r['value']['rectanglelabels']}")
            if "polygonlabels" in r["value"]:
                print(f"    polygonlabels: {r['value']['polygonlabels']}")