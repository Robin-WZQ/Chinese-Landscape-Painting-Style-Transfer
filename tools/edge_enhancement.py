import cv2 as cv
import numpy as np

def enhancement(img):
    '''图像增强算法：直方图均衡化'''
    # 计算原图中出现的最小灰度级和最大灰度级
    # 使用函数计算
    Imin, Imax = cv.minMaxLoc(img)[:2]
    # 使用numpy计算
    # Imax = np.max(img)
    # Imin = np.min(img)
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * img + b
    out = out.astype(np.uint8)
    return out
