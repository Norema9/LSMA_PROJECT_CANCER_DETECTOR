import cv2
import numpy as np
import random
import torch
import imutils
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.utils import draw_keypoints
from torchvision.io import read_image
import torchvision.transforms as transforms
import os

def crop_img(img):
    """
    Finds the extreme points on the image and crops the rectangular out of them.

    Parameters:
    - img (numpy.ndarray): Input image.

    Returns:
    - numpy.ndarray: Cropped image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    
    return new_img


def load_image(image):
    """
    Load an image using OpenCV.

    Parameters:
    - image (str): Path to the image.

    Returns:
    - numpy.ndarray: Loaded image.
    """
    return cv2.imread(image)

def image_info(image):
    """
    Get information about the image.

    Parameters:
    - image (str): Path to the image.

    Returns:
    - dict: Image information (width, height, channels).
    """
    image_orig = cv2.imread(image)
    output = {"width": image_orig.shape[1], "height": image_orig.shape[0], "channels": image_orig.shape[2]}
    return output

def show_image(image):
    """
    Display the image using OpenCV.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - None
    """
    image_orig = cv2.imread(image)
    cv2.imshow(image_orig)

def save_image(save_direct, image, image_name, extension):
    """
    Save the image to a file.

    Parameters:
    - save_direct (str): Directory where the image will be saved.
    - image (numpy.ndarray): Input image.
    - image_name (str): Name of the image.
    - extension (str): Image file extension.

    Returns:
    - None
    """
    # save the image -- OpenCV handles converting filetypes automatically
    cv2.imwrite(os.path.join(save_direct, image_name + "." + extension), image)

def draw_line(image=None, start_point=(0, 0), end_point=None, color=(0, 255, 0)):
    """
    Draw a line on the image.

    Parameters:
    - image (numpy.ndarray, optional): Input image. If None, a new canvas will be created.
    - start_point (tuple): Coordinates of the starting point (x, y).
    - end_point (tuple): Coordinates of the ending point (x, y).
    - color (tuple): Line color in BGR format.

    Returns:
    - numpy.ndarray: Image with the drawn line.
    """
    if image == None:
        # initialize our canvas as a 300x300 with 3 channels, Red, Green, and Blue, with a black background
        shape = (300, 300, 3)
        canvas = np.zeros(shape, dtype="uint8")
    else:
        canvas = image.copy()
        shape = image.shape
    if end_point == None:
        end_point = (shape[0], shape[1])

    # draw a green line from the top-left corner of our canvas to the bottom-right
    cv2.line(canvas, start_point, end_point, color)
    return canvas

def draw_rectangle(image=None, sart_corner=(0, 0), end_corner=None, color=(0, 255, 0), thick=2, filled=False):
    """
    Draw a rectangle on the image.

    Parameters:
    - image (numpy.ndarray, optional): Input image. If None, a new canvas will be created.
    - start_corner (tuple): Coordinates of the starting corner (x, y).
    - end_corner (tuple): Coordinates of the ending corner (x, y).
    - color (tuple): Rectangle color in BGR format.
    - thick (int): Thickness of the rectangle border.
    - filled (bool): Whether the rectangle should be filled.

    Returns:
    - numpy.ndarray: Image with the drawn rectangle.
    """
    if image == None:
        # initialize our canvas as a 300x300 with 3 channels, Red, Green, and Blue, with a black background
        shape = (300, 300, 3)
        canvas = np.zeros(shape, dtype="uint8")
    else:
        canvas = image.copy()
        shape = image.shape
    if end_corner == None:
        end_corner = (shape[0], shape[1])
    if filled:
        thick = -1

    # draw a green line from the top-left corner of our canvas to the bottom-right
    cv2.rectangle(canvas, sart_corner, end_corner, color, thick)
    return canvas

def draw_circle(image=None, center=None, radius=None, color=(0, 255, 0)):
    """
    Draw a circle on the image.

    Parameters:
    - image (numpy.ndarray, optional): Input image. If None, a new canvas will be created.
    - center (tuple): Coordinates of the center (x, y).
    - radius (int): Radius of the circle.
    - color (tuple): Circle color in BGR format.

    Returns:
    - numpy.ndarray: Image with the drawn circle.
    """
    if image == None:
        # initialize our canvas as a 300x300 with 3 channels, Red, Green, and Blue, with a black background
        shape = (300, 300, 3)
        canvas = np.zeros(shape, dtype="uint8")
    else:
        canvas = image.copy()
        shape = image.shape
    if radius == None:
        radius = min(canvas.shape[1]//4, canvas.shape[0]//4)
    if center == None:
        center = (canvas.shape[1] // 2, canvas.shape)