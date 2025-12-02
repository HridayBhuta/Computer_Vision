import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe as mp
from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import sys

RES_X, RES_Y = 640, 480
FPS = 30

SMOOTHING_FACTOR = 0.25
FRAME_MARGIN = 0.15

CLICK_TRIGGER_DIST = 0.06
CLICK_RELEASE_DIST = 0.08

class FingerMouse:
    def __init__(self):
        self.setup_mediapipe()
        self.setup_mouse()
        self.setup_realsense()
        self.last_x, self.last_y = 0, 0
        self.click_state = False
        self.screen_w, self.screen_h = self.get_screen_dim()

    def get_screen_dim(self):
        try:
            monitor = get_monitors()[0]
            print(f"Monitor Detected: {monitor.width}x{monitor.height}")
            return monitor.width, monitor.height
        except Exception:
            print("Warning: Could not detect monitor. Defaulting to 1920x1080.")
            return 1920, 1080

    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def setup_mouse(self):
        self.mouse = Controller()

    def setup_realsense(self):
        print("Initializing RealSense Camera...")
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, RES_X, RES_Y, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, RES_X, RES_Y, rs.format.bgr8, FPS)
            
            profile = self.pipeline.start(config)
            
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            print("Camera started successfully.")
        except Exception as e:
            print(f"Error starting RealSense: {e}")
            sys.exit(1)

    def map_coords_to_screen(self, x_norm, y_norm):
        """
        Maps normalized camera coordinates (0.0-1.0) to screen coordinates.
        Includes a 'margin' so you can reach screen edges easily.
        """
        x_clamped = np.clip(x_norm, FRAME_MARGIN, 1 - FRAME_MARGIN)
        y_clamped = np.clip(y_norm, FRAME_MARGIN, 1 - FRAME_MARGIN)
        
        x_mapped = (x_clamped - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)
        y_mapped = (y_clamped - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)

        screen_x = int(x_mapped * self.screen_w)
        screen_y = int(y_mapped * self.screen_h)
        return screen_x, screen_y

    def draw_depth_bar(self, image, depth):
        """Draws a visual indicator of finger depth and click threshold."""
        h, w = image.shape[:2]
        
        bar_w, bar_h = 30, 200
        x_start, y_start = w - 50, h // 2 - 100
        
        max_vis_depth = 0.20 
        fill_ratio = 1.0 - np.clip(depth / max_vis_depth, 0, 1)
        fill_h = int(fill_ratio * bar_h)

        trigger_y = y_start + int((1 - CLICK_TRIGGER_DIST/max_vis_depth) * bar_h)
        
        color = (0, 255, 0) if self.click_state else (0, 0, 255)

        cv2.rectangle(image, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (50, 50, 50), 2)
        cv2.rectangle(image, (x_start, y_start + bar_h - fill_h), (x_start + bar_w, y_start + bar_h), color, -1)
        cv2.line(image, (x_start - 5, trigger_y), (x_start + bar_w + 5, trigger_y), (0, 255, 255), 2)
        cv2.putText(image, "Click", (x_start - 45, trigger_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def run(self):
        print("\n--- Controls ---")
        print("Point with RIGHT Index Finger.")
        print(f"Push forward (<{CLICK_TRIGGER_DIST*100}cm) to CLICK.")
        print(f"Pull back (>{CLICK_RELEASE_DIST*100}cm) to RELEASE.")
        print("Press 'ESC' to quit.")

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue

                bg_image = np.asanyarray(color_frame.get_data())
                img_flipped = cv2.flip(bg_image, 1)
                img_h, img_w, _ = img_flipped.shape

                img_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        
                        if handedness.classification[0].label != "Right":
                            continue
                        tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        target_x, target_y = self.map_coords_to_screen(tip.x, tip.y)
                        
                        smooth_x = self.last_x + (target_x - self.last_x) * SMOOTHING_FACTOR
                        smooth_y = self.last_y + (target_y - self.last_y) * SMOOTHING_FACTOR
                        
                        self.mouse.position = (int(smooth_x), int(smooth_y))
                        self.last_x, self.last_y = smooth_x, smooth_y

                        raw_x = int((1.0 - tip.x) * img_w)
                        raw_y = int(tip.y * img_h)

                        if 0 <= raw_x < img_w and 0 <= raw_y < img_h:
                            dist = depth_frame.get_distance(raw_x, raw_y)
                            
                            if dist > 0:
                                self.draw_depth_bar(img_flipped, dist)

                                if not self.click_state and dist < CLICK_TRIGGER_DIST:
                                    self.mouse.press(Button.left)
                                    self.click_state = True
                                    cv2.circle(img_flipped, (int(tip.x*img_w), int(tip.y*img_h)), 15, (0, 255, 0), -1)
                                    
                                elif self.click_state and dist > CLICK_RELEASE_DIST:
                                    self.mouse.release(Button.left)
                                    self.click_state = False

                        self.mp_draw.draw_landmarks(img_flipped, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cv2.imshow('RealSense Finger Mouse', img_flipped)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FingerMouse()
    app.run()