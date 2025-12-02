import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_and_preprocess_images(left_image_path, right_image_path):
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    if left_img is None or right_img is None:
        print("Image is not correctly loaded")
    return left_img, right_img

def compute_disparity(left_img, right_img, disparity_range=64):
    height, width = left_img.shape
    disparity_map = np.zeros_like(left_img, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            best_offset = 0
            best_difference = float('inf')

            for offset in range(disparity_range):
                x_right = x - offset
                if x_right < 0:
                    break
                
                difference = abs(int(left_img[y, x]) - int(right_img[y, x_right]))
                
                if difference < best_difference:
                    best_difference = difference
                    best_offset = offset
            
            disparity_map[y, x] = best_offset

    return disparity_map

def smooth_disparity_map(disparity_map, ksize=5):
    return cv2.GaussianBlur(disparity_map, (ksize, ksize), 0)

def apply_color_map(disparity_map):
    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)
    colored_disparity_map = cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_INFERNO)
    return colored_disparity_map

def plot_disparity_map(colored_disparity_map):
    plt.imshow(colored_disparity_map)
    plt.title('Disparity Map with Color')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate a stereo disparity map from left and right images.')
    parser.add_argument('--left_image', type=str, help='Path to the left image')
    parser.add_argument('--right_image', type=str, help='Path to the right image')
    args = parser.parse_args()

    left_img, right_img = load_and_preprocess_images(args.left_image, args.right_image)
    disparity_map = compute_disparity(left_img, right_img, disparity_range=64)
    smoothed_disparity_map = smooth_disparity_map(disparity_map, ksize=5)
    colored_disparity_map = apply_color_map(smoothed_disparity_map)
    plot_disparity_map(colored_disparity_map)


# python Stereo_Disparity.py path_to_left_image.png path_to_right_image.png