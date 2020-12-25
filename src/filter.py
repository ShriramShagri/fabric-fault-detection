import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def filter(path, ):
    if os.path.exists(path):
        img = cv2.imread(path)
    else:
        raise Exception("Path Doesn't exist")

    img1 = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # # sure background area
    # sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)

    # # Marker labelling
    # ret, markers = cv2.connectedComponents(sure_fg)

    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers+1

    # # Now, mark the region of unknown with zero
    # markers[unknown==255] = 0

    # markers = cv2.watershed(img,markers)
    # img[markers == -1] = [255,0,0]

    return img, opening

if __name__ == "__main__":
    img, img1 = filter('C:\\Users\\Shagri\\Desktop\\DIP Mini Project\\Dataset\\Fabric22.jpg')
    plt.subplot(121),plt.imshow(cv2.cvtColor(img , cv2.COLOR_BGR2RGB)),plt.title('a'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)),plt.title('b'),plt.xticks([]),plt.yticks([])
    plt.show()