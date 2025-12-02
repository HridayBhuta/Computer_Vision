import cv2
import numpy as np
import time

# Adjust these if your red cloth isn't being detected
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

KERNEL_OPEN = np.ones((5, 5), np.uint8)
KERNEL_DILATE = np.ones((3, 3), np.uint8)

def capture_background(cap, countdown=3):
    """
    Captures the static background frame with a visual countdown.
    """
    print("Capturing background. Please move out of the frame!")
    background = None
    
    for i in range(countdown, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_disp = np.flip(frame, axis=1)
            cv2.putText(frame_disp, f"Capturing Background in {i}...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Invisibility Cloak', frame_disp)
            cv2.waitKey(1)
            background = frame

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

    background = capture_background(cap)
    if background is None:
        print("Error: Could not capture background.")
        return

    print("Background captured! You can now use the cloak.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = np.flip(frame, axis=1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask = mask1 + mask2

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, KERNEL_DILATE)

        mask_inv = cv2.bitwise_not(mask)
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        current_background = cv2.bitwise_and(frame, frame, mask=mask_inv)

        final_output = cv2.add(cloak_area, current_background)
        cv2.putText(final_output, "Press 'q' to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Invisibility Cloak', final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_invisibility_cloak()