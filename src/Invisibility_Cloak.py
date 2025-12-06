import cv2
import numpy as np
import time

# Global variables for calibration
calibrated = False
samples = []

def pick_color(event, x, y, flags, param):
    """
    Mouse callback function to capture HSV values on click/drag.
    """
    global samples
    if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
        hsv_frame = param
        if y < hsv_frame.shape[0] and x < hsv_frame.shape[1]:
            pixel = hsv_frame[y, x]
            samples.append(pixel)

def get_color_ranges(samples, std_dev_mult=2.5):
    """
    Calculates LOWER and UPPER HSV bounds based on the mean and std dev of samples.
    Handles the special case where Hue wraps around 0/180 (like Red).
    """
    if not samples:
        return [], []

    data = np.array(samples)
    
    H = data[:, 0]
    S = data[:, 1]
    V = data[:, 2]

    is_wrapping = False
    if np.max(H) - np.min(H) > 90:
        if np.any(H < 20) and np.any(H > 160):
            is_wrapping = True
            H = np.where(H < 90, H + 180, H)

    mean_h, std_h = np.mean(H), np.std(H)
    mean_s, std_s = np.mean(S), np.std(S)
    mean_v, std_v = np.mean(V), np.std(V)

    lower_s = np.clip(mean_s - std_dev_mult * std_s, 0, 255)
    upper_s = np.clip(mean_s + std_dev_mult * std_s, 0, 255)
    lower_v = np.clip(mean_v - std_dev_mult * std_v, 0, 255)
    upper_v = np.clip(mean_v + std_dev_mult * std_v, 0, 255)

    lower_bounds = []
    upper_bounds = []

    if is_wrapping:
        l_h = mean_h - std_dev_mult * std_h
        u_h = mean_h + std_dev_mult * std_h

        if l_h < 180 and u_h > 180:
             lower_bounds.append(np.array([l_h, lower_s, lower_v]))
             upper_bounds.append(np.array([180, upper_s, upper_v]))
             lower_bounds.append(np.array([0, lower_s, lower_v]))
             upper_bounds.append(np.array([u_h - 180, upper_s, upper_v]))
        elif l_h >= 180:
             lower_bounds.append(np.array([l_h - 180, lower_s, lower_v]))
             upper_bounds.append(np.array([u_h - 180, upper_s, upper_v]))
        else:
             lower_bounds.append(np.array([l_h, lower_s, lower_v]))
             upper_bounds.append(np.array([u_h, upper_s, upper_v]))

    else:
        l_h = np.clip(mean_h - std_dev_mult * std_h, 0, 179)
        u_h = np.clip(mean_h + std_dev_mult * std_h, 0, 179)
        
        lower_bounds.append(np.array([l_h, lower_s, lower_v]))
        upper_bounds.append(np.array([u_h, upper_s, upper_v]))

    return lower_bounds, upper_bounds

def calibrate_color(cap):
    """
    Runs the calibration loop. Returns list of lower bounds and list of upper bounds.
    """
    global samples
    samples = []
    
    print("--- CALIBRATION MODE ---")
    print("1. Hold up your cloth.")
    print("2. Click and drag mouse over the cloth to sample colors.")
    print("3. Press 'c' to calculate and save.")
    print("4. Press 'r' to reset samples.")

    window_name = "Calibration"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        cv2.setMouseCallback(window_name, pick_color, hsv)

        disp_frame = frame.copy()
        cv2.putText(disp_frame, f"Samples: {len(samples)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp_frame, "Click on cloth. Press 'c' to done.", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(window_name, disp_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(samples) > 10:
                print("Calculating color range...")
                lowers, uppers = get_color_ranges(samples)
                print(f"Ranges calculated: {len(lowers)} pair(s).")
                for l, u in zip(lowers, uppers):
                    print(f"  L: {l.astype(int)}  U: {u.astype(int)}")
                cv2.destroyWindow(window_name)
                return lowers, uppers
            else:
                print("Not enough samples! Click more.")
        elif key == ord('r'):
            samples = []
            print("Samples reset.")
        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            exit()

    return [], []

def capture_background(cap, countdown=3):
    """
    Captures the static background frame with a visual countdown.
    """
    print("Capturing background. Please move out of the frame!")
    
    for i in range(countdown, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_disp = np.flip(frame, axis=1)
            frame_disp = np.ascontiguousarray(frame_disp) 

            cv2.putText(frame_disp, f"Capturing Background in {i}...", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Invisibility Cloak', frame_disp)
            cv2.waitKey(1)

    print("Capturing final background...")
    backgrounds = []
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            backgrounds.append(frame)
    
    if not backgrounds:
        return None

    avg_bg = np.median(backgrounds, axis=0).astype(np.uint8)
    return np.flip(avg_bg, axis=1)

def run_invisibility_cloak():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    lower_bounds, upper_bounds = calibrate_color(cap)
    
    if not lower_bounds:
        print("Calibration failed or cancelled. Using default RED.")
        lower_bounds = [np.array([0, 120, 70]), np.array([170, 120, 70])]
        upper_bounds = [np.array([10, 255, 255]), np.array([180, 255, 255])]

    background = capture_background(cap)
    if background is None:
        print("Error: Could not capture background.")
        return

    print("Background captured! You can now use the cloak.")
    print("Press 'q' to quit.")

    KERNEL_OPEN = np.ones((5, 5), np.uint8)
    KERNEL_DILATE = np.ones((3, 3), np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.flip(frame, axis=1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for low, high in zip(lower_bounds, upper_bounds):
            l = np.array(low, dtype=np.uint8)
            u = np.array(high, dtype=np.uint8)
            mask += cv2.inRange(hsv, l, u)
            
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL_DILATE)

        mask_inv = cv2.bitwise_not(mask)
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        current_background = cv2.bitwise_and(frame, frame, mask=mask_inv)

        final_output = cv2.add(cloak_area, current_background)

        cv2.imshow('Invisibility Cloak', final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_invisibility_cloak()