"""
# @Time    :  2020/9/8
# @Author  :  Jimou Chen
"""
import cv2
import matplotlib.pyplot as plt

'''读入图片后可以像操作矩阵那样操作图片'''


def draw_img(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


img = cv2.imread('image.jpg')
'''对图片进行修改'''

# 高、宽、通道数
h, w, c = img.shape
print(h, w, c)

# 截取其中的1/4
img1 = img[0: h // 2, 0:w // 2]
draw_img(img1)

# 把左上角区域变蓝
img[0: h // 2, 0:w // 2] = (0, 0, 225)
draw_img(img)
