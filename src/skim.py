import sys
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer

# get filename and sigma value from command line
# filename = sys.argv[1]
# sigma = float(sys.argv[2])
filename = '20.jpg'
sigma = float(5)

# read and display the original image
image = skimage.io.imread(fname=filename)
viewer = skimage.viewer.ImageViewer(image)
# viewer.show()

# blur and grayscale before thresholding
blur = skimage.color.rgb2gray(image)
blur = skimage.filters.gaussian(blur, sigma=sigma)

# perform adaptive thresholding
t = skimage.filters.threshold_otsu(blur)
mask = blur < t

viewer = skimage.viewer.ImageViewer(mask)
# viewer.show()

# use the mask to select the "interesting" part of the image
sel = np.zeros_like(image)
sel[mask] = image[mask]

# display the result
viewer = skimage.viewer.ImageViewer(sel)
viewer.show()