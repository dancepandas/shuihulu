import json

with open("D:/chengs/9.project/shuihulu/datasets/project-3-at-2026-05-14-10-37-b3c9ac5a.json", encoding="utf-8") as f:
    data = json.load(f)

for i, item in enumerate(data[:5]):
    file_upload = item.get("file_upload", "")
    image_path = item.get("data", {}).get("image", "")
    print(f"Item {i}: file_upload={file_upload}, image={image_path}")