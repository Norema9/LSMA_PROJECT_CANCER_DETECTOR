from skimage import feature
import numpy as np
import cv2
import imutils
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import os


class ResNetFeature:
    def __init__(self) -> None:
        # Load ResNet50 model, excluding top layer
        self.model = ResNet50(include_top=False, pooling='avg')

    def describe(self, image_path):
        img = image.load_img(image_path)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        features = self.model.predict(img_data)
        return features.flatten()


class ColorChannelStatistic:
    def __init__(self):
        pass
    
    def describe(self, image_path):
        """
        Extracts color channel statistics from an image.

        Args:
            image_path (str): The path to the input image.

        Returns:
            numpy.ndarray: Concatenated mean and standard deviation of each color channel.
        """
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Compute the mean and standard deviation for each color channel
        (means, stds) = cv2.meanStdDev(image)

        # Concatenate the mean and standard deviation arrays and flatten them
        features = np.concatenate([means, stds]).flatten()

        return features
    

class LabHistogram:
    def __init__(self, bins=[8, 8, 8]):
        """
        Initialize the LabHistogram object.

        Args:
            bins (list): Number of bins for each dimension of the histogram. Default is [8, 8, 8].
        """
        # Store the number of bins for the histogram
        self.bins = bins

    def describe(self, image_path, mask=None):
        """
        Compute the L*a*b* color histogram of an image.

        Args:
            image_path (str): Path to the input image.
            mask (numpy.ndarray, optional): Mask to apply to the image. Default is None.

        Returns:
            numpy.ndarray: Flattened L*a*b* color histogram.
        """
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to the L*a*b* color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Compute the histogram
        hist = cv2.calcHist([lab], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        hist = cv2.normalize(hist, hist).flatten()

        return hist


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        """
        Initialize the LocalBinaryPatterns object.

        Args:
            numPoints (int): Number of points to use for LBP.
            radius (int): Radius of the circle to consider for LBP.
        """
        # Store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image_path, eps=1e-7):
        """
        Compute the Local Binary Pattern (LBP) histogram of an image.

        Args:
            image_path (str): Path to the input image.
            eps (float, optional): Small value to avoid division by zero when normalizing histogram. Default is 1e-7.

        Returns:
            numpy.ndarray: Normalized histogram of Local Binary Patterns.
        """
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the Local Binary Pattern representation of the image
        lbp = feature.local_binary_pattern(gray, self.numPoints, self.radius, method="uniform")

        # Calculate the histogram of patterns
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, int(lbp.max()) + 3), range=(0, int(lbp.max()) + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # Return the histogram of Local Binary Patterns as a 1D array
        return hist


class HOG:
    def __init__(self):
        """
        Initialize the HOG object.
        """
        pass

    def describe(self, image_path):
        """
        Compute the Histogram of Oriented Gradients (HOG) descriptor of an image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            numpy.ndarray: HOG descriptor.
        """
        # Read the image using OpenCV and convert it to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Apply edge detection
        # edged = imutils.auto_canny(gray)

        # # Find contours in the edge map, keeping only the largest one
        # # which is presumed to be the car logo
        # cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # c = max(cnts, key=cv2.contourArea)

        # # Extract the logo of the car and resize it to a canonical width
        # # and height
        # (x, y, w, h) = cv2.boundingRect(c)
        # logo = gray[y:y + h, x:x + w]
        # logo = cv2.resize(logo, (200, 100))

        # Extract Histogram of Oriented Gradients from the logo
        H = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                        transform_sqrt=True, block_norm="L1")

        return H