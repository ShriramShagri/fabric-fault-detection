import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
ddepth = cv.CV_16S
kernel_size = 3
img = cv.imread(os.path.join(os.getcwd(), 'Dataset', '15.jpg'))
original = img.copy()
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(gray, (55, 55), 0)
# median = cv.medianBlur(gray,15)
# ddepth = cv.CV_8U 
# kernel_size = 3

# dst = cv.Laplacian(gray, ddepth, ksize=kernel_size)
# abs_dst = cv.convertScaleAbs(dst)
# dst = cv.Laplacian(img, ddepth, ksize=kernel_size)
# abs_dst = cv.convertScaleAbs(dst)
ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv.THRESH_BINARY,23,2)
# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
#             cv.THRESH_BINARY,25,2)



# kernel = np.ones((3,3),np.uint8)
# opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

# ret1, img1 = cv.threshold(thresh, 200, 255, cv.THRESH_BINARY)
# ret2, img2 = cv.threshold(thresh, 255, 255, cv.THRESH_BINARY)
# double_threshold = img1 - img2

# find normalized_histogram, and its cumulative distribution function

# # Emboss effect
# # level around 1 with hnot more than 0.4 dont go very low as 0.5


# level = 1.2
# image = cv2.imread('Image.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# kernel = cv2.getGaborKernel((5, 5), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# # kernel = np.array([[-1, -1, -1],[-1 ,8,-1],[-1,-1,-1]])*level

# img = cv2.filter2D(image, -1, kernel)
# # img = cv2.GaussianBlur(image, (5, 5), 0)

# sure_bg = cv.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)

plt.subplot(121),plt.imshow(cv.cvtColor(original , cv.COLOR_BGR2RGB)),plt.title('a'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(cv.cvtColor(th3, cv.COLOR_BGR2RGB)),plt.title('b'),plt.xticks([]),plt.yticks([])
plt.show()