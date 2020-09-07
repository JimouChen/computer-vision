"""
# @Time    :  2020/9/7
# @Author  :  Jimou Chen
"""
import cv2
import matplotlib.pyplot as plt


def draw_img(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


img = cv2.imread('image.jpg')
print(img.shape)
draw_img(img)

# 转换为RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
draw_img(img)

# 保存
cv2.imwrite('new_img.jpg', img)
