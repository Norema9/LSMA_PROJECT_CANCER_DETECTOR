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
	Finds the extreme points on the image and crops the rectangular out of them
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
     return cv2.imread(image)

def image_info(image) -> dict:
    image_orig = cv2.imread(image)
    output = {"width": image_orig.shape[1], "height": image_orig.shape[0], "channels": image_orig.shape[2]}
    return output

def show_image(image) -> None:
    image_orig = cv2.imread(image)
    cv2.imshow(image_orig)

def save_image(save_direct, image, image_name, extension):
    # save the image -- OpenCV handles converting filetypes automatically
    cv2.imwrite(os.path.join(save_direct, image_name + "." + extension), image)

def draw_line(image = None, start_point:tuple = (0, 0), end_point:tuple = None, color:tuple = (0, 255, 0)):
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

def draw_rectangle(image = None, sart_corner:tuple = (0, 0), end_corner:tuple = None, color:tuple = (0, 255, 0), thick: int = 2, filled:bool = False):
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

def draw_circle(image = None, center:tuple = None, radius:int = None, color:tuple = (0, 255, 0)):
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
        center = (canvas.shape[1] // 2, canvas.shape[0] // 2)

    # draw a green line from the top-left corner of our canvas to the bottom-right
    cv2.circle(canvas, center, radius, color)
    return canvas

def translate(image_orig, rigth = None, left = None, up = None, down = None):
    # Copy Image
    image = image_orig.copy()
    aux = False
    shifts = []
    if rigth != None:
        aux = True
        shifts.append([1, 0, rigth])
    if left != None:
        aux = True
        shifts.append([1, 0, -left])
    if up != None:
        aux = True
        shifts.append([0, 1, -up])
    if down != None:
        aux = True
        shifts.append([0, 1, down])
    if not aux:
        raise print("ArgumentError: no direction has been provided for the translation")

    # NOTE: Translating (shifting) an image is given by a NumPy matrix in the form: [[1, 0, shiftX], [0, 1, shiftY]]
    # You simply need to specify how many pixels you want to shift the image in the X and Y direction -- let's translate the image 25 pixels to the right and 50 pixels down
    M = np.float32(shifts)
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


# # finally, let's use our helper function in imutils to shift the image down 100 pixels
# shifted = imutils.translate(image, 0, 100)
# print("Shifted Down")
# cv2_imshow(shifted)

def rotate(image_orig, angle, center=None, scale=1.0):
    # Copy Image
    image = image_orig.copy()
    
    # grab the dimensions of the image
    (h, w) = image.shape[:2]
    
    # if the center is None, initialize it as the center of the image
    if center is None:
        center = (w // 2, h // 2)
    
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated

def resize(image_orig, width=None, height=None, inter=cv2.INTER_AREA):
    # Copy Image
    image = image_orig.copy()
    
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]

    # if the width is None, then resize the image based on the height
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # if the height is None, then resize the image based on the width
    else:
        r = width / float(w)
        dim = (width, int(h * r))
        dim = (width, int(h * r))

    # perform the resizing
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def flip(image_orig, flip_code):
    # Copy Image
    image = image_orig.copy()
    
    # Perform the flipping
    flipped = cv2.flip(image, flip_code)
    
    return flipped

def crop(image_orig, start_y, end_y, start_x, end_x):
    # Copy Image
    image = image_orig.copy()
    
    # Perform cropping using NumPy array slices
    cropped = image[start_y:end_y, start_x:end_x]
    
    return cropped

def vary_intensity(image_orig, alpha=1.0, beta=0):
    # Copy Image
    image = image_orig.copy()
    
    # Apply brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return adjusted

def apply_mask(image_orig, mask_type, mask_params):
    # Create a blank mask
    mask = np.zeros(image_orig.shape[:2], dtype="uint8")
    
    # Generate the mask based on the selected type
    if mask_type == 'rectangle':
        cv2.rectangle(mask, mask_params['start_point'], mask_params['end_point'], 255, -1)
    elif mask_type == 'circle':
        cv2.circle(mask, mask_params['center'], mask_params['radius'], 255, -1)
    else:
        raise ValueError("Invalid mask type. Use 'rectangle' or 'circle'.")

    # Apply the mask to the original image
    masked = cv2.bitwise_and(image_orig, image_orig, mask=mask)
    return masked

def split_and_merge_channels(image_orig):
    # Split channels
    (B, G, R) = cv2.split(image_orig)

    # Show each channel individually
    print("RED, GREEN, BLUE")
    cv2.imshow("Red Channel", R)
    cv2.imshow("Green Channel", G)
    cv2.imshow("Blue Channel", B)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Merge the channels back together
    merged = cv2.merge([B, G, R])
    print("Merged")
    cv2.imshow("Merged Image", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Visualize each channel in color
    zeros = np.zeros(image_orig.shape[:2], dtype="uint8")
    print("RED, GREEN, BLUE")
    cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
    cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
    cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
