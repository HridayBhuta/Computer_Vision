import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from pynput.mouse import Button, Controller
from screeninfo import get_monitors

# --- Setup ---
# MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# pynput Mouse Controller
mouse = Controller()

# Screen Info (for mapping coordinates)
screen = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = screen.width, screen.height
print(f"Screen dimensions: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# --- Mouse Control Configuration ---
SMOOTHING_FACTOR = 0.2 # 0.0 = no smoothing, 1.0 = no movement
last_mouse_x, last_mouse_y = 0, 0

# --- Click Configuration ---
is_clicking = False
CLICK_DEPTH_THRESHOLD = 0.1 # 30 cm from camera
# This state tracks if we are *in* a click (button is down)
click_in_progress = False

print("--- Finger Mouse & Click Started ---")
print("Move your RIGHT index finger to control the mouse.")
print(f"Push your finger closer than {CLICK_DEPTH_THRESHOLD}m to click.")
print("Press ESC in the video window to quit.")

try:
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1) as hands: # Only track one hand for performance
        
        while True:
            # --- Get Frames ---
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue

            # --- Process Image ---
            color_image = np.asanyarray(color_frame.get_data())
            # Flip horizontally for a mirror view
            color_image = cv2.flip(color_image, 1) 
            
            # Get image dimensions for pixel conversion
            h, w, _ = color_image.shape
            
            # Convert BGR (OpenCV) to RGB (MediaPipe)
            # We process the *flipped* image so coordinates match the mirror view
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            results = hands.process(image_rgb)

            # --- Detect Hand and Move Mouse ---
            if results.multi_hand_landmarks:
                # We only care about the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                hand_type = handedness.classification[0].label

                # Only control mouse with RIGHT hand
                if hand_type == "Right":
                    # --- Add text label for hand ---
                    wrist_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    text_x = int(wrist_landmark.x * w) - 30
                    text_y = int(wrist_landmark.y * h) - 20
                    cv2.putText(color_image, "Right Hand", 
                                (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                    # Get Index Fingertip (landmark 8)
                    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    
                    # --- 1. MOUSE POSITION (X, Y) ---
                    
                    # *** MIRRORING FIX: ***
                    # We no longer invert the x-axis with (1 - tip.x)
                    # Because the image fed to MediaPipe is already flipped,
                    # tip.x (e.g., 0.9) now correctly means "right side of the screen".
                    target_x = tip.x * SCREEN_WIDTH
                    target_y = tip.y * SCREEN_HEIGHT
                    
                    # Apply smoothing
                    mouse_x = (target_x * SMOOTHING_FACTOR) + (last_mouse_x * (1 - SMOOTHING_FACTOR))
                    mouse_y = (target_y * SMOOTHING_FACTOR) + (last_mouse_y * (1 - SMOOTHING_FACTOR))
                    
                    mouse.position = (int(mouse_x), int(mouse_y))
                    
                    last_mouse_x, last_mouse_y = mouse_x, mouse_y

                    # --- 2. MOUSE CLICK (Z) ---
                    # Get 3D (z) coordinate in meters
                    # Convert normalized (x,y) to pixel (x,y)
                    # We MUST use the *unflipped* x coordinate for the depth map
                    unflipped_x = w - int(tip.x * w)
                    pixel_y = int(tip.y * h)

                    current_z = 0.0
                    # Check if pixel is valid before getting distance
                    if 0 < unflipped_x < w and 0 < pixel_y < h:
                        # Use the unflipped_x to sample the *original* depth frame
                        current_z = depth_frame.get_distance(unflipped_x, pixel_y)

                    # Check for a valid depth reading (not 0)
                    if current_z > 0.0: 
                        # If finger is "pushed" (closer than threshold)
                        if current_z < CLICK_DEPTH_THRESHOLD:
                            if not click_in_progress:
                                print(f"CLICK PRESS (Depth: {current_z:.2f}m)")
                                mouse.press(Button.left)
                                click_in_progress = True
                        # If finger is "pulled back" (father than threshold)
                        elif current_z > CLICK_DEPTH_THRESHOLD:
                            if click_in_progress:
                                print(f"CLICK RELEASE (Depth: {current_z:.2f}m)")
                                mouse.release(Button.left)
                                click_in_progress = False
                    # If depth reading is invalid (e.g., 0.0), release the click
                    else:
                        if click_in_progress:
                            print("CLICK RELEASE (Depth Invalid)")
                            mouse.release(Button.left)
                            click_in_progress = False

                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(
                        color_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # --- Show Video Feed ---
            cv2.imshow('Index Finger Mouse - RealSense', color_image)
            
            if cv2.waitKey(5) & 0xFF == 27: # Press ESC to quit
                break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    # Failsafe: Make sure the mouse button is released when the app closes
    if click_in_progress:
        mouse.release(Button.left)
    print("--- Index Finger Mouse Stopped ---")