import numpy as np
import cv2
from scipy import ndimage
from scipy import linalg
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import pdb 
    

def convolution(image, kernel):
    '''
    Performs convolution along x and y axis, based on kernel size.
    Assumes input image is 1 channel (Grayscale)
    Inputs: 
      image: H x W x C shape numpy array (C=1)
      kernel: K_H x K_W shape numpy array (for example, 3x1 for 1 dimensional filter for y-component)
    Returns:
      H x W x C Image convolved with kernel
    '''



def gaussian_kernel(size=3, sigma=1):
    '''
    Creates Gaussian Kernel (1 Dimensional)
    Inputs: 
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D gaussian kernel
    '''

def gaussian_first_derivative_kernel(size=3, sigma=1):
    '''
    Creates 1st derviative gaussian Kernel (1 Dimensional)
    Inputs: 
      size: width of the filter
      sigma: standard deviation
    Returns a 1xN shape 1D 1st derivative gaussian kernel
    '''


def non_max_supression(det, phase):
    '''
    Performs non-maxima supression for given magnitude and orientation.
    Returns output with nms applied. Also return a colored image based on gradient direction for maximum value.
    '''


def DFS(img):
    '''
    If pixel is linked to a strong pixel in a local window, make it strong as well.
    Called iteratively to make all strong-linked pixels strong.
    '''
    

                    
def hysteresis_thresholding(img, low_ratio, high_ratio):

    
if __name__ == '__main__':
    # Initialize values
    # You can choose any sigma values like 1, 0.5, 1.5, etc


    # Read the image in grayscale mode using opencv

    # Create a gaussian kernel 1XN matrix

    # Convolution of G and I

    # Get the First Derivative Kernel


    # Derivative of Gaussian Convolution

    # Convert derivative result to 0-255 for display.
    # Need to scale from 0-1 to 0-255.
    #abs_grad_x = (( (I_xx - np.min(I_xx)) / (np.max(I_xx) - np.min(I_xx)) ) * 255.).astype(np.uint8)  
    #abs_grad_y = (( (I_yy - np.min(I_yy)) / (np.max(I_yy) - np.min(I_yy)) ) * 255.).astype(np.uint8)


    # Compute magnitude

    # Compute orientation

    
    # Compute non-max suppression

    #Compute thresholding and then hysteresis
