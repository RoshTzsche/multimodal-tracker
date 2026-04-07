################
## RoshTzsche ##
################
import cv2
import mediapipe as mp
import numpy as np
import os
import platform
import time
import urllib.request
from actions import ActionController


class MultiModalSystem:
    def __init__(self):
        print("[System] Initializing MediaPipe Tasks API...")
        self._ensure_models_exist()

        # ── MediaPipe Tasks API handles ────────────────────────────────────────
        BaseOptions = mp.tasks.BaseOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Hand Landmarker — detects up to 1 hand in VIDEO mode for low latency
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        options_hands = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./models/hand_landmarker.task'),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options_hands)

        # Face Landmarker — detects up to 1 face in VIDEO mode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        options_face = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./models/face_landmarker.task'),
            running_mode=self.VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options_face)

        # Standard 21-point hand skeleton connection table
        # Each tuple is a pair of landmark indices to connect with a line
        self.hand_connections = [
            (0, 1),  (1, 2),  (2, 3),  (3, 4),   # Thumb
            (0, 5),  (5, 6),  (6, 7),  (7, 8),   # Index
            (5, 9),  (9, 10), (10, 11),(11, 12),  # Middle
            (9, 13), (13, 14),(14, 15),(15, 16),  # Ring
            (13, 17),(17, 18),(18, 19),(19, 20),  # Pinky
            (0, 17),                               # Palm base
        ]

        # Action / overlay controller
        self.controller = ActionController()

    # ── Model management ────────────────────────────────────────────────────────

    def _ensure_models_exist(self):
        """Downloads MediaPipe .task model files if they are not already present."""
        os.makedirs('./models', exist_ok=True)
        models = {
            './models/hand_landmarker.task': (
                'https://storage.googleapis.com/mediapipe-models/'
                'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
            ),
            './models/face_landmarker.task': (
                'https://storage.googleapis.com/mediapipe-models/'
                'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
            ),
        }
        for path, url in models.items():
            if not os.path.exists(path):
                print(f"[System] Downloading model: {os.path.basename(path)} ...")
                urllib.request.urlretrieve(url, path)
                print(f"[System] Saved to {path}")

    # ── Pure OpenCV drawing helpers ─────────────────────────────────────────────

    def _draw_hand_skeleton(self, image, landmarks, w: int, h: int):
        """
        Draws the 21-point hand skeleton using pure OpenCV calls.
        Bypasses the legacy mediapipe.solutions drawing utilities to avoid
        compatibility issues with the newer Tasks API result schema.
        """
        # Convert normalized coordinates to pixel positions
        points = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)  # Red joint dots

        # Draw bones between connected landmark pairs
        for idx1, idx2 in self.hand_connections:
            if idx1 < len(points) and idx2 < len(points):
                cv2.line(image, points[idx1], points[idx2], (0, 255, 0), 2)  # Green lines

    def _draw_face_dots(self, image, landmarks, w: int, h: int):
        """
        Draws a lightweight dot mesh over the 468 face landmarks.
        Using 1-pixel dots is fast and avoids hardcoding the full tessellation graph.
        """
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (cx, cy), 1, (255, 255, 0), -1)  # Cyan dots

    # ── Gesture classification ───────────────────────────────────────────────────

    def _is_finger_up(self, lm_list, finger_tip_idx: int, finger_pip_idx: int) -> bool:
        """
        Determines whether a finger is extended by comparing squared Euclidean
        distances from the wrist — avoids the cost of sqrt() on every frame.

        A finger is "up" when the tip is farther from the wrist than the PIP joint:
            d²(wrist, tip) > d²(wrist, pip)
        """
        wrist = lm_list[0]
        tip   = lm_list[finger_tip_idx]
        pip   = lm_list[finger_pip_idx]

        d2_tip = (tip[1] - wrist[1]) ** 2 + (tip[2] - wrist[2]) ** 2
        d2_pip = (pip[1] - wrist[1]) ** 2 + (pip[2] - wrist[2]) ** 2
        return d2_tip > d2_pip

    def classify_hand(self, lm_list) -> str:
        """
        Classifies the hand pose into one of:
        THUMB_UP, THUMB_DOWN, FIST, OPEN_PALM, PEACE, POINT, UNKNOWN
        """
        if not lm_list:
            return "UNKNOWN"

        thumb_tip = lm_list[4]
        thumb_mcp = lm_list[2]

        # Check all four non-thumb fingers (tip / PIP landmark pairs)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        fingers_up = [
            self._is_finger_up(lm_list, tip, pip)
            for tip, pip in zip(finger_tips, finger_pips)
        ]
        total_up = fingers_up.count(True)

        # All four fingers down → classify by thumb y-position
        if total_up == 0:
            threshold = 20
            if thumb_tip[2] < thumb_mcp[2] - threshold:
                return "THUMB_UP"
            elif thumb_tip[2] > thumb_mcp[2] + threshold:
                return "THUMB_DOWN"
            else:
                return "FIST"

        if total_up >= 4:
            return "OPEN_PALM"

        # index + middle up → peace / victory sign
        if fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            return "PEACE"

        # only index up → pointing
        if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
            return "POINT"

        return "UNKNOWN"

    # ── Face expression classification ──────────────────────────────────────────

    def _get_euclidean_distance(self, p1, p2) -> float:
        """Returns the Euclidean distance between two (x, y) points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _calculate_ear(self, landmarks, indices, w: int, h: int) -> float:
        """
        Computes the Eye Aspect Ratio (EAR) for a set of 6 eye landmarks.

            EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Returns 0.0 if the horizontal distance is zero (degenerate case).
        """
        coords = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
        d_v1 = self._get_euclidean_distance(coords[1], coords[5])
        d_v2 = self._get_euclidean_distance(coords[2], coords[4])
        d_h  = self._get_euclidean_distance(coords[0], coords[3])
        return (d_v1 + d_v2) / (2.0 * d_h) if d_h > 0 else 0.0

    def classify_face(self, landmarks, w: int, h: int) -> str:
        """
        Classifies the current facial expression into one of:
        SURPRISED, WINK_LEFT, WINK_RIGHT, SMILE, NEUTRAL

        Uses three analytic geometry metrics:
          • MAR (Mouth Aspect Ratio)  — detects open mouth / surprise
          • EAR (Eye Aspect Ratio)    — detects winks
          • Smile ratio               — mouth width vs face width
        """
        # ── Mouth Aspect Ratio ─────────────────────────────────────────────────
        top   = (landmarks[13].x * w,  landmarks[13].y * h)
        bot   = (landmarks[14].x * w,  landmarks[14].y * h)
        left  = (landmarks[61].x * w,  landmarks[61].y * h)
        right = (landmarks[291].x * w, landmarks[291].y * h)

        mouth_h = self._get_euclidean_distance(top, bot)
        mouth_w = self._get_euclidean_distance(left, right)

        if mouth_w == 0:
            return "NEUTRAL"

        mar = mouth_h / mouth_w
        if mar > 0.45:
            return "SURPRISED"

        # ── Eye Aspect Ratio (EAR) ─────────────────────────────────────────────
        # MediaPipe 468-point mesh landmark indices for each eye (6 points each)
        left_eye_indices  = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33,  160, 158, 133, 144, 153]

        ear_left  = self._calculate_ear(landmarks, left_eye_indices,  w, h)
        ear_right = self._calculate_ear(landmarks, right_eye_indices, w, h)

        blink_threshold = 0.2
        if ear_left  < blink_threshold and ear_right > blink_threshold:
            return "WINK_LEFT"
        if ear_right < blink_threshold and ear_left  > blink_threshold:
            return "WINK_RIGHT"

        # ── Smile ratio ────────────────────────────────────────────────────────
        face_left  = (landmarks[234].x * w, landmarks[234].y * h)
        face_right = (landmarks[454].x * w, landmarks[454].y * h)
        face_width = self._get_euclidean_distance(face_left, face_right)

        if face_width > 0:
            smile_ratio = mouth_w / face_width
            # Wide mouth + closed lips = smile
            if smile_ratio > 0.42 and mar < 0.3:
                return "SMILE"

        return "NEUTRAL"

    # ── Overlay compositor ───────────────────────────────────────────────────────

    def overlay_image(self, background, overlay, x: int, y: int):
        """
        Alpha-composites a BGRA overlay image onto the background at position (x, y).
        Handles boundary clamping automatically; returns the background unchanged
        if either image is None or the clipped region has zero area.
        """
        if overlay is None or background is None:
            return background

        h_ov, w_ov = overlay.shape[:2]
        h_bg, w_bg = background.shape[:2]

        # Clamp coordinates to stay within the background bounds
        x = max(x, 0)
        y = max(y, 0)
        w_ov = min(w_ov, w_bg - x)
        h_ov = min(h_ov, h_bg - y)

        if w_ov <= 0 or h_ov <= 0:
            return background

        overlay_crop = overlay[:h_ov, :w_ov]
        bg_slice     = background[y:y + h_ov, x:x + w_ov]

        b, g, r, a = cv2.split(overlay_crop)
        alpha     = a / 255.0          # Normalized alpha mask  [0.0 – 1.0]
        alpha_inv = 1.0 - alpha        # Inverse mask for background blending

        # Blend each BGR channel using the alpha mask
        for c in range(3):
            bg_slice[:, :, c] = (
                alpha * overlay_crop[:, :, c] + alpha_inv * bg_slice[:, :, c]
            )

        background[y:y + h_ov, x:x + w_ov] = bg_slice
        return background

    # ── Camera initialization ───────────────────────────────────────────────────

    def _initialize_camera(self, camera_index: int = 0):
        """
        Opens the webcam using the best available backend for the current OS:
          • Linux   → V4L2
          • Windows → DirectShow (DSHOW)
          • macOS   → AVFoundation

        Falls back to the generic OpenCV backend if the OS-specific one fails.
        Change `camera_index` to 1, 2, etc. for external USB cameras.
        """
        current_os = platform.system()
        print(f"[Hardware] OS detected: {current_os}  |  Camera index: {camera_index}")

        backend_map = {
            "Linux":   cv2.CAP_V4L2,
            "Windows": cv2.CAP_DSHOW,
            "Darwin":  cv2.CAP_AVFOUNDATION,
        }

        cap = None
        if current_os in backend_map:
            cap = cv2.VideoCapture(camera_index, backend_map[current_os])

        # Generic fallback
        if cap is None or not cap.isOpened():
            print("[Hardware] OS-specific backend unavailable — falling back to generic.")
            cap = cv2.VideoCapture(camera_index)

        return cap

    # ── Main loop ───────────────────────────────────────────────────────────────

    def run(self):
        """Starts the webcam capture loop and runs hand + face inference each frame."""
        cap = self._initialize_camera(camera_index=0)

        if not cap.isOpened():
            print("[Critical Error] Could not open camera. Check connections and permissions.")
            return

        # Request 30 FPS and automatic white balance
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        window_name = 'Multimodal Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        print("[System] Tracker running. Press ESC to quit.")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # Transient frame drop — keep going
                continue

            # Mirror the image so it feels like a selfie camera
            image = cv2.flip(image, 1)
            h, w, _ = image.shape

            # Convert BGR → RGB and wrap in a MediaPipe Image container
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Monotonically increasing timestamp required by VIDEO running mode
            timestamp_ms = int(time.time() * 1000)

            # Run both detectors on the current frame
            res_hands = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            res_face  = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_gesture    = "UNKNOWN"
            face_expression = "NEUTRAL"

            # ── Hand processing ────────────────────────────────────────────────
            if res_hands and res_hands.hand_landmarks:
                for hand_landmarks in res_hands.hand_landmarks:
                    self._draw_hand_skeleton(image, hand_landmarks, w, h)
                    # Build flat landmark list: [id, px_x, px_y]
                    lm_list = [
                        [idx, int(lm.x * w), int(lm.y * h)]
                        for idx, lm in enumerate(hand_landmarks)
                    ]
                    hand_gesture = self.classify_hand(lm_list)

            # ── Face processing ────────────────────────────────────────────────
            if res_face and res_face.face_landmarks:
                for face_landmarks in res_face.face_landmarks:
                    self._draw_face_dots(image, face_landmarks, w, h)
                    face_expression = self.classify_face(face_landmarks, w, h)

            # ── Combo overlay ──────────────────────────────────────────────────
            overlay_img = self.controller.get_overlay_image(hand_gesture, face_expression)
            if overlay_img is not None:
                image = self.overlay_image(image, overlay_img, w - 220, 20)
                cv2.putText(
                    image, "COMBO!",
                    (w - 200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )

            # ── HUD status bar ─────────────────────────────────────────────────
            status = f"Gesture: {hand_gesture}  |  Face: {face_expression}"
            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, image)

            # ESC key exits the loop
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[System] Tracker stopped cleanly.")


if __name__ == "__main__":
    app = MultiModalSystem()
    app.run()
