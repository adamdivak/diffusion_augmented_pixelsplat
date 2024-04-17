import cv2
import numpy as np
import blur_detector

def opencv_blur_detection(image):
    """
    Apply blur detection using OpenCV functions.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Binary mask indicating regions of potential blur.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate edges to close holes
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    _, contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find biggest contour
    biggest_contour = max(contours, key=cv2.contourArea)
    
    # Draw contours on mask
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [biggest_contour], -1, (255), -1)
    
    # Fill in holes
    inverted_mask = cv2.bitwise_not(mask)
    _, contours, _ = cv2.findContours(inverted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_contours = [cnt for cnt in contours if cv2.contourArea(cnt) < 20000]
    cv2.drawContours(mask, small_contours, -1, (255), -1)
    
    # Opening + median blur to smooth mask
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    
    return mask


def transformers_blur_detection(image):
    """
    Apply blur detection using transformers library.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Blurred map indicating regions of potential blur.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return blur_detector.detectBlur(gray_image, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3, show_progress=True)


def custom_blur_detection(image, sigma: int = 5, min_abs: float = 0.5):
    """
    Apply custom blur detection algorithm.

    Parameters:
        image (numpy.ndarray): Input image.
        threshold (int): Threshold value for Laplacian operator.
        sigma (int): Standard deviation for blurring.
        min_abs (float): Minimum absolute value for log operation.

    Returns:
        numpy.ndarray: Blurred map indicating regions of potential blur.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Laplacian operator
    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    
    # Absolute values and thresholding
    abs_image = np.abs(blur_map).astype(np.float32)
    abs_image[abs_image < min_abs] = min_abs
    
    # Log operation
    abs_image = np.log(abs_image)
    
    # Blurring
    cv2.blur(abs_image, (sigma, sigma))
    
    # Median blur
    return cv2.medianBlur(abs_image, sigma)
