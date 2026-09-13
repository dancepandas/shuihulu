import json

with open("D:/chengs/9.project/shuihulu/datasets/project-3-at-2026-05-14-10-50-b3c9ac5a.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total items: {len(data)}")

missing = [
    "DJI_20260115104108_0001_V",
    "DJI_20260115141601_0015_V",
    "DJI_20260115141604_0016_V",
    "DJI_20260115141642_0018_V",
    "DJI_20260115141645_0019_V",
    "DJI_20260115141650_0021_V",
    "DJI_20260115141804_0025_V",
    "DJI_20260115141819_0026_V",
    "DJI_20260115141826_0027_V",
    "DJI_20260115141831_0028_V",
    "DJI_20260115141853_0029_V",
    "DJI_20260115141909_0030_V",
    "DJI_20260115141921_0031_V",
    "DJI_20260115142007_0032_V",
    "DJI_20260115142807_0033_V",
    "DJI_20260115142856_0035_V",
    "DJI_20260115143418_0041_V",
    "DJI_20260115143526_0046_V",
    "DJI_20260115143754_0058_V",
    "DJI_20260115143758_0059_V",
    "DJI_20260115143808_0060_V",
    "DJI_20260115143812_0061_V",
    "DJI_20260115143815_0062_V",
    "DJI_20260115143820_0063_V",
]

found = set()
for item in data:
    fu = item.get("file_upload", "")
    orig = fu.split("-", 1)[-1] if "-" in fu else fu
    stem = orig.rsplit(".", 1)[0]
    if stem in missing:
        found.add(stem)

print(f"Missing images found in new export: {len(found)}")
not_found = set(missing) - found
if not_found:
    print(f"Still missing: {len(not_found)}")
    for s in sorted(not_found):
        print(f"  {s}.JPG")