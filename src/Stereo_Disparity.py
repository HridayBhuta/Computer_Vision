import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import time

def load_and_preprocess_images(left_image_path, right_image_path):
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    
    if left_img is None:
        raise FileNotFoundError(f"Could not load left image: {left_image_path}")
    if right_img is None:
        raise FileNotFoundError(f"Could not load right image: {right_image_path}")
        
    return left_img, right_img

def compute_disparity_naive(left_img, right_img, disparity_range=64):
    """
    O(N^3) implementation. Extremely slow, effectively 'reference' only.
    """
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

def compute_disparity_fast(left_img, right_img, disparity_range=64):
    """
    Vectorized NumPy implementation. 
    Calculates cost volume by shifting images instead of looping pixels.
    """
    height, width = left_img.shape
    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)
    
    cost_volume = np.full((height, width, disparity_range), np.inf, dtype=np.float32)

    for d in range(disparity_range):
        if d == 0:
            diff = np.abs(left_img - right_img)
            cost_volume[:, :, d] = diff
        else:
            diff = np.abs(left_img[:, d:] - right_img[:, :-d])
            cost_volume[:, d:, d] = diff

    disparity_map = np.argmin(cost_volume, axis=2)
    return disparity_map.astype(np.uint8)

def compute_disparity_opencv(left_img, right_img, disparity_range=64):
    """
    Implementation using OpenCV's highly optimized Block Matching.
    """
    num_disp = disparity_range
    if num_disp % 16 != 0:
        num_disp = ((num_disp // 16) + 1) * 16
        print(f"Adjusted disparity range to {num_disp} for OpenCV compatibility.")

    stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=15)
    
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    
    disparity[disparity < 0] = 0
    return disparity.astype(np.uint8)

def smooth_disparity_map(disparity_map, ksize=5):
    return cv2.GaussianBlur(disparity_map, (ksize, ksize), 0)

def apply_color_map(disparity_map):
    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)
    colored_disparity_map = cv2.applyColorMap(disparity_map_normalized, cv2.COLORMAP_INFERNO)
    return colored_disparity_map

def plot_disparity_map(colored_disparity_map, method_name, elapsed_time):
    plt.imshow(cv2.cvtColor(colored_disparity_map, cv2.COLOR_BGR2RGB))
    plt.title(f'Method: {method_name} | Time: {elapsed_time:.4f}s')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a stereo disparity map from left and right images.')
    parser.add_argument('left_image', type=str, help='Path to the left image')
    parser.add_argument('right_image', type=str, help='Path to the right image')
    parser.add_argument('--range', type=int, default=64, help='Maximum disparity range (default: 64)')
    parser.add_argument('--method', type=str, choices=['naive', 'vectorized', 'opencv'], default='vectorized',
                        help='Algorithm to use: "naive" (slow loop), "vectorized" (numpy fast), "opencv" (optimized BM)')
    
    args = parser.parse_args()

    try:
        left_img, right_img = load_and_preprocess_images(args.left_image, args.right_image)
        
        print(f"Computing disparity using '{args.method}' method...")
        start_time = time.time()
        
        if args.method == 'naive':
            print("Warning: Naive method is O(N^3) and extremely slow.")
            disparity_map = compute_disparity_naive(left_img, right_img, disparity_range=args.range)
        elif args.method == 'opencv':
            disparity_map = compute_disparity_opencv(left_img, right_img, disparity_range=args.range)
        else:
            disparity_map = compute_disparity_fast(left_img, right_img, disparity_range=args.range)
            
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Computation complete in {elapsed:.4f} seconds.")

        smoothed_disparity_map = smooth_disparity_map(disparity_map, ksize=5)
        colored_disparity_map = apply_color_map(smoothed_disparity_map)
        plot_disparity_map(colored_disparity_map, args.method, elapsed)
        
    except Exception as e:
        print(f"Error: {e}")