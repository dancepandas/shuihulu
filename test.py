import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图片 (请确保图片路径正确，这里假设图片名为 'input.jpg')
# 如果你是在本地运行，请把 'input.jpg' 换成你的图片路径
# 为了演示，我这里使用你提供的图片链接（如果网络允许）或者你需要手动指定路径
image_path = 'datasets/hyacinth_seg/images/train/DJI_20260115141508_0011_V.JPG'
try:
    img = cv2.imread(image_path)
    if img is None:
        # 如果读取失败，提示用户
        print("未找到图片，请确保图片路径正确。")
        # 这里为了演示效果，我创建一个模拟的空白图，实际使用时请替换
        img = np.zeros((1000, 1500, 3), dtype=np.uint8) + 100
except:
    print("请替换为你的图片路径")

h, w, _ = img.shape

# 2. 定义标注数据 (归一化坐标)
# 格式: [x1, y1, x2, y2, ...]
annotations = [
    # 主体大片水草
    [0.000, 0.210, 0.100, 0.205, 0.200, 0.200, 0.300, 0.195, 0.400, 0.190,
     0.500, 0.185, 0.600, 0.180, 0.700, 0.175, 0.800, 0.175, 0.900, 0.175,
     1.000, 0.175, 1.000, 0.250, 0.980, 0.350, 0.950, 0.450, 0.920, 0.550,
     0.880, 0.650, 0.820, 0.750, 0.750, 0.850, 0.680, 0.920, 0.620, 0.950,
     0.580, 0.900, 0.550, 0.800, 0.530, 0.700, 0.520, 0.600, 0.510, 0.500,
     0.480, 0.400, 0.450, 0.300, 0.420, 0.200, 0.400, 0.100, 0.380, 0.050,
     0.350, 0.000, 0.300, 0.000],

    # 水面零星团块 A
    [0.020, 0.650, 0.040, 0.640, 0.060, 0.660, 0.050, 0.680, 0.030, 0.670],
    # 水面零星团块 B
    [0.120, 0.680, 0.150, 0.670, 0.180, 0.700, 0.160, 0.730, 0.130, 0.720],
    # 水面零星团块 C
    [0.190, 0.840, 0.210, 0.830, 0.220, 0.860, 0.200, 0.870]
]

# 3. 绘制
overlay = img.copy()

for pts in annotations:
    # 将归一化坐标转换为像素坐标
    points = []
    for i in range(0, len(pts), 2):
        x = int(pts[i] * w)
        y = int(pts[i + 1] * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # 绘制半透明填充 (红色)
    cv2.fillPoly(overlay, [points], (0, 0,
                                     255))  # Blue channel is 0, Green 0, Red 255 -> Red color in OpenCV is (0,0,255) wait, OpenCV is BGR. So (0,0,255) is Red.

    # 绘制轮廓 (绿色)
    cv2.polylines(img, [points], True, (0, 255, 0), 2)

# 混合图层
alpha = 0.4
result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# 4. 显示结果
plt.figure(figsize=(20, 15))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("YOLOv8-seg Annotation: Waterweeds/Algae")
plt.axis('off')
plt.tight_layout()
plt.show()
