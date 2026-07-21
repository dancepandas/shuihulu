"""验证 SDK mask2rle / encode_rle 能正确 round-trip"""
import numpy as np
from label_studio_sdk.converter.brush import mask2rle, decode_rle

# 构造一个已知 mask (H=6, W=5), 中间有个矩形
H, W = 6, 5
mask = np.zeros((H, W), dtype=np.uint8)
mask[2:4, 1:4] = 1
print("orig mask:\n", mask)

rle = mask2rle(mask)
print("rle len:", len(rle), "max:", max(rle), "min:", min(rle))
print("all <=255 ?", all(0 <= v <= 255 for v in rle))

# 解码: decode_rle 返回 1d 长度 = H*W*4
decoded = decode_rle(rle)
print("decoded len:", len(decoded))
img4 = np.reshape(decoded, [H, W, 4])
alpha = img4[:, :, 3]
print("decoded alpha:\n", alpha)
print("MATCH:", np.array_equal(mask, alpha))
