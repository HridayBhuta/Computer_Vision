import cv2
import numpy as np
import mediapipe as mp
from pynput.mouse import Button, Controller
from screeninfo import get_monitors
import sys

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False

RES_X, RES_Y = 640, 480
FPS = 30

SMOOTHING_FACTOR = 0.25
FRAME_MARGIN = 0.15

RS_CLICK_TRIGGER = 0.06
RS_CLICK_RELEASE = 0.08
WC_CLICK_TRIGGER = 0.10
WC_CLICK_RELEASE = 0.08

class FingerMouse:
    def __init__(self):
        self.setup_mediapipe()
        self.setup_mouse()
        self.use_realsense = False
        self.cap = None
        self.setup_camera()
        
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

    def setup_camera(self):
        """
        Attempts to initialize RealSense. Falls back to Webcam on failure.
        """
        if RS_AVAILABLE:
            print("Attempting to connect to Intel RealSense...")
            try:
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, RES_X, RES_Y, rs.format.z16, FPS)
                config.enable_stream(rs.stream.color, RES_X, RES_Y, rs.format.bgr8, FPS)
                
                self.pipeline.start(config)
                
                align_to = rs.stream.color
                self.align = rs.align(align_to)
                self.use_realsense = True
                print("Success: Intel RealSense connected.")
                return
            except Exception as e:
                print(f"RealSense initialization failed: {e}")
        else:
            print("pyrealsense2 library not found.")

        print("Falling back to standard Webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_X)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_Y)
        print("Webcam started successfully.")

    def map_coords_to_screen(self, x_norm, y_norm):
        x_clamped = np.clip(x_norm, FRAME_MARGIN, 1 - FRAME_MARGIN)
        y_clamped = np.clip(y_norm, FRAME_MARGIN, 1 - FRAME_MARGIN)
        
        x_mapped = (x_clamped - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)
        y_mapped = (y_clamped - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)

        screen_x = int(x_mapped * self.screen_w)
        screen_y = int(y_mapped * self.screen_h)
        return screen_x, screen_y

    def draw_depth_bar(self, image, depth, trigger_thresh, max_range=0.20):
        """
        Draws a visual indicator of finger depth and click threshold.
        Adapts to the current mode (RealSense or Webcam).
        """
        h, w = image.shape[:2]
        
        bar_w, bar_h = 30, 200
        x_start, y_start = w - 50, h // 2 - 100
        
        fill_ratio = 1.0 - np.clip(depth / max_range, 0, 1)
        fill_h = int(fill_ratio * bar_h)

        trigger_ratio = np.clip(trigger_thresh / max_range, 0, 1)
        trigger_y = y_start + int((1 - trigger_ratio) * bar_h)
        
        color = (0, 255, 0) if self.click_state else (0, 0, 255)

        cv2.rectangle(image, (x_start, y_start), (x_start + bar_w, y_start + bar_h), (50, 50, 50), 2)
        cv2.rectangle(image, (x_start, y_start + bar_h - fill_h), (x_start + bar_w, y_start + bar_h), color, -1)
        cv2.line(image, (x_start - 5, trigger_y), (x_start + bar_w + 5, trigger_y), (0, 255, 255), 2)
        cv2.putText(image, "Click", (x_start - 45, trigger_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def run(self):
        print("\n--- Controls ---")
        print("Point with RIGHT Index Finger.")
        if self.use_realsense:
            print(f"Push forward (<{RS_CLICK_TRIGGER*100}cm) to CLICK.")
        else:
            print("Webcam Mode: Bend finger or push hand forward to CLICK.")
        print("Press 'ESC' to quit.")

        try:
            while True:
                if self.use_realsense:
                    frames = self.pipeline.wait_for_frames()
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    if not depth_frame or not color_frame: continue
                    bg_image = np.asanyarray(color_frame.get_data())
                else:
                    ret, bg_image = self.cap.read()
                    depth_frame = None
                    if not ret: break

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

                        current_depth = 0.0
                        trigger_val = 0.0
                        release_val = 0.0
                        
                        if self.use_realsense and depth_frame:
                            raw_x = int((1.0 - tip.x) * img_w)
                            raw_y = int(tip.y * img_h)
                            
                            if 0 <= raw_x < img_w and 0 <= raw_y < img_h:
                                current_depth = depth_frame.get_distance(raw_x, raw_y)
                                trigger_val = RS_CLICK_TRIGGER
                                release_val = RS_CLICK_RELEASE
                        else:
                            current_depth = -tip.z if tip.z < 0 else 0
                            trigger_val = WC_CLICK_TRIGGER
                            release_val = WC_CLICK_RELEASE

                        if current_depth > 0:
                            self.draw_depth_bar(img_flipped, current_depth, trigger_val)
                            is_click_action = False
                            is_release_action = False

                            if self.use_realsense:
                                is_click_action = (current_depth < trigger_val)
                                is_release_action = (current_depth > release_val)
                            else:
                                is_click_action = (current_depth > trigger_val)
                                is_release_action = (current_depth < release_val)

                            if not self.click_state and is_click_action:
                                self.mouse.press(Button.left)
                                self.click_state = True
                                cv2.circle(img_flipped, (int(tip.x*img_w), int(tip.y*img_h)), 15, (0, 255, 0), -1)
                                
                            elif self.click_state and is_release_action:
                                self.mouse.release(Button.left)
                                self.click_state = False

                        self.mp_draw.draw_landmarks(img_flipped, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                cv2.imshow('Finger Mouse', img_flipped)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            if self.use_realsense:
                self.pipeline.stop()
            elif self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FingerMouse()
    app.run()