# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from math import *
import os

# Create an images folder to put our images in
try:
    os.mkdir('images')
except:
    pass


def display(imglist, size=5, shape=True):
    """
    This function will take in a list of matrices and display them as images
    Parameters
    ----------
    imglist : list
        List of matrices (as ndarray) to display as images

    size : int, optional
        An image's display size in inches. The default is 5.

    shape : bool, optional
        Whether or not the dimensions of an image should be displayed above it. 
        The default is True.
    """

    cols = len(imglist)
    fig = plt.figure(figsize=(size*cols, size*cols))
    for i in range(0, cols):
        a = fig.add_subplot(1, cols, i+1)
        if len(imglist[i].shape) > 2:
            subfig = plt.imshow(imglist[i], vmin=0.0, vmax=1.0)
        else:
            subfig = plt.imshow(imglist[i], cmap="gray", vmin=0.0, vmax=1.0)
        subfig.axes.get_xaxis().set_visible(False)
        subfig.axes.get_yaxis().set_visible(False)
        if shape == True:
            a.set_title(str(imglist[i].shape))
    plt.show()


def multiply_mask(image: np.array, mask: np.array) -> np.array:
    """
    Combines an image with a black and white mask
    Parameters
    ----------
    image : np.array
        Original image as an array

    mask : np.array
        Black and white mask image as an array
    Returns
    -------
    img : np.array
        Combined image
    """
    img = np.zeros((image.shape[0], image.shape[1], 3))

    for line in range(image.shape[0]):
        for column in range(image.shape[1]):
            # We apply the formula given by multiplying the image's pixels
            # with the mask's pixels
            img[line, column] = image[line, column]*mask[line, column]

    return img


def twist(image: np.array, rho: float) -> np.array:
    """
    Apply a twist effect to an image
    Parameters
    ----------
    image : np.array
        Original image

    rho : float
        Represents the twisting factor
    Returns
    -------
    twist : np.array
        Image with twisted filter applied
    """

    twist = np.zeros((image.shape[0], image.shape[1], 3))
    # Get center coordinates
    centerj = image.shape[1]/2
    centeri = image.shape[0]/2
    for line in range(image.shape[0]):
        for column in range(image.shape[1]):

            # Apply the formula to find alpha
            # rho and sqrt come from the `math` library
            alpha = rho*sqrt((column-centerj)**2+(line-centeri)**2)
            # Apply the Matrix equation to get the new coordinates
            # np.add : Matrix addition
            # np.dot : Matrix multiplication
            coords = np.add(
                np.dot(
                    [
                        [
                            cos(-alpha),
                            -sin(-alpha)
                        ],
                        [
                            sin(-alpha),
                            cos(-alpha)
                        ]
                    ],
                    [
                        column-centerj,
                        line-centeri
                    ]),
                [centerj, centeri]
            ).astype(int)
            # Make sure our coordinates are within bounds
            if coords[1] < image.shape[0] and coords[1] >= 0 and coords[0] < image.shape[1] and coords[0] >= 0:
                twist[line, column] = image[coords[1], coords[0]]

    return twist


def clamp(img: np.array) -> np.array:
    """
    Clamps a 3D array containing rgb integer values in the 0-255 range
    to rgb float values in the 0.0-1.0 range
    Parameters
    ----------
    img : np.array
        The original image as an array of pixels.
    Returns
    -------
    clamped_img : np.array
        An array of clamped pixels representing the original image
    """

    # Create an image array of the same size as our img parameter, filled with
    # zero values for each pixel
    clamped_img = np.zeros((img.shape[0], img.shape[1], 3))
    # Loop through each pixel (line, column)
    for line in range(img.shape[0]):
        for column in range(img.shape[1]):
            # Divide by 255 each pixel
            clamped_img[line, column] = img[line, column]/255

    return clamped_img


# Our pixels are declared as rgb values in a list
a = [1.0, 0.0, 0.0]
b = [0.0, 1.0, 0.0]
c = [0.0, 0.0, 1.0]
d = [0.0, 0.0, 0.0]
# To help us with matrix equation functions
# we'll use numpy's array
image = np.array(
    [
        [a, b],
        [c, d]
    ]
)
# we then return a list of our images to
# the previously declared display function
# display([image])


image_as_array = plt.imread("images/cormoran.jpg")
image = clamp(image_as_array)
# And of course we can output that array as an image again
# display([image_as_array])


display([image, twist(image, 0.01), twist(image, 0.02), twist(image, 0.05)])
