import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import os


import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

def add_noise(image):
    """Adds Gaussian noise to an image."""
    row, col, ch = image.shape
    mean = 0
    sigma = 5
    gauss = np.random.normal(mean, sigma, (row, col, ch)).reshape(row, col, ch)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def apply_high_pass_filter(image):
    """Applies a high-pass filter to an image by subtracting a blurred version from the original."""
    low_pass = gaussian_filter(image, sigma=0.5)
    high_pass = image - low_pass
    return high_pass

def apply_low_pass_filter(image):
    """Applies a low-pass filter to an image, smoothing it by blurring."""
    blurred_image = gaussian_filter(image, sigma=1)
    return blurred_image

def apply_distortion(image):
    """Applies a perspective distortion to an image, slightly skewing its geometry."""
    src_points = np.float32([[0, 0], [image.shape[1]-1, 0], [0, image.shape[0]-1], [image.shape[1]-1, image.shape[0]-1]])
    dst_points = np.float32([[0, 0], [image.shape[1]-1, 0], [image.shape[1]*0.05, image.shape[0]*0.95], [image.shape[1]*0.95, image.shape[0]*0.95]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    distorted_image = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))
    return distorted_image

def apply_twisting(image):
    """Applies a twisting effect to an image by rotating it around its center."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 10, 1.0)
    twisted_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return twisted_image

def apply_linear_wave(image):
    """Applies a horizontal wave distortion to an image, using a sinusoidal offset pattern."""
    rows, cols = image.shape[:2]
    img_output = np.zeros(image.shape, dtype=image.dtype)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(5.0 * np.sin(2 * np.pi * i / 360))
            new_j = (j + offset_x) % cols
            img_output[i, j] = image[i, new_j]
    return img_output


def process_image(image_path, save_path):
    image = cv2.imread(image_path)
    functions = [add_noise, apply_high_pass_filter, apply_low_pass_filter, apply_distortion, apply_twisting, apply_linear_wave]
    selected_functions = random.sample(functions, 3)  # Randomly select 3 different functions
    for func in selected_functions:
        image = func(image)
    cv2.imwrite(save_path, image)


if __name__ == '__main__':
    img_1, img_2, img_3 = "img_1.jpeg", "img_2.JPG", "img_3.jpeg"
    os.makedirs("results", exist_ok=True)
    process_image(img_1, "results/processed_img_4.jpeg")
    process_image(img_2, "results/processed_img_5.JPG")
    process_image(img_3, "results/processed_img_6.jpeg")
