import cv2
import numpy as np

# https://blog.csdn.net/saltriver/article/details/78987170?utm_source=app
# https://www.jb51.net/article/168142.htm
img = cv2.imread("d:/7.jpg")
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img2 = np.copy(img)
img2 = img2.astype("float")
sp = img.shape
print(sp)
Robot = np.zeros(sp)
for i in range(sp[0]):
    for j in range(sp[1]):
        if i + 1 <= sp[0] - 1 and j + 1 <= sp[1] - 1:
            tmp = abs(int(img2[i][j]) - int(img2[i + 1][j + 1])) + abs(int(img2[i][j + 1]) - int(img2[i + 1][j]))
            Robot[i, j] = tmp
#
sharp = Robot
#sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))
sharp = sharp.astype("uint8")



cv2.imshow("Robot", sharp)




# Roberts 算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)

x = cv2.filter2D(img, cv2.CV_16S, kernelx)
y = cv2.filter2D(img, cv2.CV_16S, kernely)

# 转 uint8 ,图像融合
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts_cv2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("Robot_cv2",Roberts_cv2)
cv2.waitKey(0)
