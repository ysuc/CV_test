#CV_show

import cv2
import numpy as np
import matplotlib.pyplot as plt

#图像读取与尺寸转换
'''image = cv2.imread("C:/CV-test/1.1.jpg")  #读取图像
print(image)  #打印多维矩阵
plt.imshow(image)
plt.show()  #显示图片

print(image.shape)  #显示图像维度
image1=cv2.resize(image,(512,512))  #改变图像尺寸
print(image1.shape)
plt.imshow(image1)
plt.show()  #显示图片'''

#图像切片
'''image = cv2.imread("C:/CV-test/1.1.jpg")  #读取图像
image1 = image[56:342,128:512,:]  #通过切片进行切片裁剪
print(image1)  #打印多维矩阵
plt.imshow(image1)
plt.show()  #显示图片'''

#图像二值化
'''img = cv2.imread("C:/CV-test/1.3.jpg")  #读取图像
t,img1 = cv2.threshold(img,52,255,cv2.THRESH_BINARY)  #二值化阈值处理
cv2.imshow('dst',img1)
cv2.waitKey()  #按下任意键启动  
cv2.destroyAllWindows()  #显示所有窗口'''

#图像旋转
'''img = cv2.imread("C:/CV-test/1.3.jpg")  #读取图像
rows = len(img)  #获取图像像素行数
cols = len(img[0])  #获取图像像素列数
center = (rows/2,cols/2)  #获取图像中心点
M = cv2.getRotationMatrix2D(center,30,1.0)  #以图象中心为轴，逆时针旋转30°，不施加缩放
img1 = cv2.warpAffine(img,M,(cols,rows))  #按照M进行仿射变换
cv2.imshow('dst',img1)
cv2.waitKey()  #按下任意键启动  
cv2.destroyAllWindows()  #显示所有窗口'''

#图像滤波
image = cv2.imread("D:/002.jpg")  #读取图像
dst = cv2.GaussianBlur(image,(7,7),0,0)  #使用宽度为19的矩形滤波核进行中值滤波
cv2.imshow("5",dst)
cv2.waitKey()  #按下任意键启动  
cv2.destroyAllWindows()  #显示所有窗口

#图像腐蚀膨胀算法
'''image = cv2.imread("D:/_陈昊哲的桌面/研一文件/CV-test/1.4.jpg")  #读取图像   #图像读取路径不应包含中文
k = np.ones((3,3),np.uint8)  #创建3*3矩阵作为核
dst = cv2.erode(image,k)  #进行腐蚀操作
cv2.imshow("dst",dst)
cv2.waitKey()  #按下任意键启动  
cv2.destroyAllWindows()  #显示所有窗口'''