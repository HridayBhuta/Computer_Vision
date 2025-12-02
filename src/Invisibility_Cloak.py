import cv2
import numpy as np
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)
time.sleep(3)  # Allow the camera to warm up

# Capture the background frame
for i in range(30):
    ret, background = cap.read()

# Flip the background frame
background = np.flip(background, axis=1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame
    frame = np.flip(frame, axis=1)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the color to mask (e.g., red)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Define the range for the color to mask (e.g., blue)
    # lower_blue = np.array([94, 80, 2])
    # upper_blue = np.array([126, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    # lower_blue = np.array([94, 80, 2])
    # upper_blue = np.array([126, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the masks
    mask = mask1 + mask2

    # Refine the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Create an inverse mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment the cloak part
    res1 = cv2.bitwise_and(background, background, mask=mask)

    # Segment the non-cloak part
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine the results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the output
    cv2.imshow('Invisibility Cloak', final_output)

    # Break the loop on 'q' key press1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()