import os
from ..utils import *
import cv2 as cv
import matplotlib.pyplot as plt

class Preprocessor():
    """
    Used for filtering the dataset before inputting into CNN
    """    
    def __init__(self, inpath):
        """
        Preprocessor Constructor

        Args:
            inpath (str): Path of image file

        Raises:
            ImageFileNotFoundError: Raised when image is not found in the specified path
        """        
        if not os.path.exists(inpath):
            raise ImageFileNotFoundError

        # Image source
        self.src = cv.imread(inpath)

        # Convert to grayscale or back to bgr
        self.toGray = cv.COLOR_BGR2GRAY
        self.toBGR = cv.COLOR_GRAY2RGB

        # Values for adaptive tresholding
        self.adaptiveMethod = cv.ADAPTIVE_THRESH_GAUSSIAN_C
        self.tresholdType = cv.THRESH_BINARY
        self.tresholdMaxValue = 255
        self.tresholdBlockSize = 23
        self.C = 2

        # Gaussian Blur
        self.ksize = (45, 45)
        self.sigmaX = 0
    
    def filter(self):
        """
        Applies filter to the image

        Returns:
            image: Filtered Image
        """        
        self.image = self.src
        self.image = self._gray(self.image)
        self.image = self._treshold(self.image)
        self.image = self._grayTobgr(self.image)

        return self.image
    
    def display(self):
        """
        Displays filtered and original images using matplotlib
        """     
        # Apply filter Before plotting   
        self.filter()

        # plot both original and filtererd images
        plt.subplot(121),plt.imshow(cv.cvtColor(self.src , cv.COLOR_BGR2RGB)),plt.title('Original'),plt.xticks([]),plt.yticks([])
        plt.subplot(122),plt.imshow(cv.cvtColor(self.image, cv.COLOR_BGR2RGB)),plt.title('Filtered'),plt.xticks([]),plt.yticks([])
        plt.show()

    def _gray(self, img):
        """
        Convert BGR image to grayscale and apply Gaussian Blur

        Args:
            img (cv2 image): Image to be converted

        Raises:
            ImageProcessingError: Raises when failed to process the image

        Returns:
            cv2 image: filtered image
        """        
        try:
            gray = cv.cvtColor(img, self.toGray)
            blur = cv.GaussianBlur(gray, self.ksize, self.sigmaX)
        
        except:
            raise ImageProcessingError("Error in Preprocessor._gray method")

        else:
            return blur
    
    def _grayTobgr(self, img):
        """
        Convert grayscale image to BGR 

        Args:
            img (cv2 image): Image to be converted

        Raises:
            ImageProcessingError: Raises when failed to process the image

        Returns:
            cv2 image: filtered image
        """   
        try:
            converted = cv.cvtColor(img, self.toBGR)
        
        except:
            raise ImageProcessingError("Error in Preprocessor._grayTobgr method")

        else:
            return converted
    
    def _treshold(self, img):
        """
        Uses Adaptive Tresholding to filter photo

        Args:
            img (cv2 image): Image to be converted(grayscale)

        Raises:
            ImageProcessingError: Raises when failed to process the image

        Returns:
            cv2 image: filtered image
        """ 
        try:
            tresholdimage = cv.adaptiveThreshold( img,
                                            self.tresholdMaxValue,
                                            self.adaptiveMethod,
                                            self.tresholdType,
                                            self.tresholdBlockSize,
                                            self.C)
        except:
            raise ImageProcessingError("Error in Preprocessor._treshold method")
        else:
            return tresholdimage