"""
# @Time    :  2020/9/8
# @Author  :  Jimou Chen
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_img(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# 创造一张黑底图片,长宽为300的3通道图片
black_img = np.zeros((300, 300, 3), dtype='uint8')
draw_img(black_img)

# 画直线
green = (0, 255, 0)
cv2.line(black_img, (0, 0), (300, 300), green, thickness=5)  # 里面是起点到终点
draw_img(black_img)

blue = (0, 0, 225)
cv2.line(black_img, (300, 0), (150, 150), blue, 8)
draw_img(black_img)

# 画矩形
red = (255, 0, 0)
cv2.rectangle(black_img, (20, 20), (60, 60), red, 2)
draw_img(black_img)
# -1是填充
cv2.rectangle(black_img, (50, 50), (100, 250), green, -1)
draw_img(black_img)

# 画圆
white = (255, 255, 255)
# 圆心位置
(rx, ry) = black_img.shape[1] // 2, black_img.shape[0] // 2
# 循环画出不同半径的圆
for r in range(0, 151, 15):
    cv2.circle(black_img, (rx, ry), r, white, 2)
draw_img(black_img)

'''随机生成圆'''
# 重新弄个黑底画布
new_img = np.zeros((300, 300, 3), dtype='uint8')
# 画10个圆
for i in range(10):
    # 随机取半径
    r = np.random.randint(5, 200)
    # 随机取颜色
    color = np.random.randint(0, 255, size=(3,)).tolist()
    # 圆心
    point = np.random.randint(0, 300, size=(2,))
    # 画有填充的
    cv2.circle(new_img, tuple(point), r, color, -1)

draw_img(new_img)
