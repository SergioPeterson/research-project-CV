# Main imports
import numpy as np
import cv2
from importlib import reload
import utils2; reload(utils2)
from utils2 import *

def threshold_img(img, channel, thres=(0, 255)):
    """
    Applies a threshold mask to the input image
    """
    img_ch = img[:,:,channel]
    if thres is None:
        return img_ch

    mask_ch = np.zeros_like(img_ch)
    mask_ch[ (thres[0] <= img_ch) & (thres[1] >= img_ch) ] = 1
    return mask_ch


def compute_hsv_white_red_binary(rgb_img):
    """
    Returns a binary thresholded image produced retaining only white and yellow elements on the picture
    The provided image should be in RGB format
    """
    hsv_img = to_hsv(rgb_img)

    # Red color thresholds
    red_hue_min = int(0 * 179)  # Lower bound for red hue
    red_hue_max = int(179)  # Upper bound for red hue (allowing some range)
    red_saturation_min = int(0.5 * 255)  # Lower bound for red saturation
    red_saturation_max = int(1.0 * 255)  # Upper bound for red saturation
    red_value_min = int(0.5 * 255)  # Lower bound for red value
    red_value_max = int(1 * 255)  # Upper bound for red value

    img_hsv_red_bin = np.zeros_like(hsv_img[:,:,0])
    img_hsv_red_bin[((hsv_img[:,:,0] >= red_hue_min) & (hsv_img[:,:,0] <= red_hue_max))
                    & ((hsv_img[:,:,1] >= red_saturation_min) & (hsv_img[:,:,1] <= red_saturation_max))
                    & ((hsv_img[:,:,2] >= red_value_min) & (hsv_img[:,:,2] <= red_value_max))] = 1

    # White color thresholds
    white_saturation_max = int(0.3 * 255)  # Adjusted saturation maximum for white
    white_value_min = int(0.9 * 255)  # Adjusted value minimum for white

    img_hsv_white_bin = np.zeros_like(hsv_img[:,:,0])
    img_hsv_white_bin[((hsv_img[:,:,1] <= white_saturation_max))
                      & ((hsv_img[:,:,2] >= white_value_min))] = 1

    # Combine both red and white binary images
    img_hls_white_red_bin = np.zeros_like(hsv_img[:,:,0])
    img_hls_white_red_bin[(img_hsv_red_bin == 1) | (img_hsv_white_bin == 1)] = 1

    return img_hls_white_red_bin



def abs_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))
    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
    return gradient_mask

def mag_sobel(gray_img, kernel_size=3, thres=(0, 255)):
    """
    Computes sobel matrix in both x and y directions, merges them by computing the magnitude in both directions
    and applies a threshold value to only set pixels within the specified range
    """
    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sxy = np.sqrt(np.square(sx) + np.square(sy))
    scaled_sxy = np.uint8(255 * sxy / np.max(sxy))

    sxy_binary = np.zeros_like(scaled_sxy)
    sxy_binary[(scaled_sxy >= thres[0]) & (scaled_sxy <= thres[1])] = 1

    return sxy_binary
def dir_sobel(gray_img, kernel_size=3, thres=(0, np.pi/2)):
    """
    Computes sobel matrix in both x and y directions, gets their absolute values to find the direction of the gradient
    and applies a threshold value to only set pixels within the specified range
    """
    sx_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sy_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))

    dir_sxy = np.arctan2(sx_abs, sy_abs)

    binary_output = np.zeros_like(dir_sxy)
    binary_output[(dir_sxy >= thres[0]) & (dir_sxy <= thres[1])] = 1

    return binary_output
def combined_sobels(sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=3, angle_thres=(0, np.pi/2)):
    sxy_direction_binary = dir_sobel(gray_img, kernel_size=kernel_size, thres=angle_thres)

    combined = np.zeros_like(sxy_direction_binary)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels
    combined[(sx_binary == 1) | ((sy_binary == 1) & (sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1

    return combined
def compute_perspective_transform_matrices(src, dst):
    """
    Returns the tuple (M, M_inv) where M represents the matrix to use for perspective transform
    and M_inv is the matrix used to revert the transformed image back to the original one
    """
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return (M, M_inv)

def perspective_transform(img, src, dst):
    """
    Applies a perspective
    """
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped



def get_combined_binary_thresholded_img(img):
    """
    Applies a combination of binary Sobel and color thresholding to an undistorted image
    Those binary images are then combined to produce the returned binary image
    """
    #Peramiter one
    # thresholded_img = threshold_img(img, 2, thres=None)
    threshold_image = threshold_img(img, 1, thres=None)

    #Peramiter two
    sobx_best = abs_sobel(threshold_image, kernel_size=15, thres=(80, 200))

    #Peramiter three
    soby_best = abs_sobel(threshold_image, x_dir=False, kernel_size=11, thres=(50, 150))

    #Peramiter four
    sobxy_best = mag_sobel(threshold_image, kernel_size=15, thres=(80, 200))

    #Peramiter five
    sobel_combined_best = combined_sobels(sobx_best, soby_best, sobxy_best, threshold_image, kernel_size=15, angle_thres=(0, np.pi/4))


    hsv_w_y_thres = compute_hsv_white_red_binary(img)
    combined_binary = np.zeros_like(hsv_w_y_thres)
    combined_binary[(sobel_combined_best == 1) | (hsv_w_y_thres == 1)] = 1


    # return sxy_combined_dir #DEBUG UNCOMMENT
    return combined_binary