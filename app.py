"""
Hand Filter Window
==================

Hold both hands up to the webcam. Your thumb and index fingertips become the
corners of a live "filter window". Touch your thumb to a finger to pick the
filter shown inside the window:

    thumb + index  -> Grayscale
    thumb + middle -> Color Scale (pinch again to step through fixed colors)
    thumb + ring   -> Blur
    thumb + pinky  -> Pixelate

Keys:  q = quit   l = toggle landmark overlay   h = clean view (filter only)
"""

import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CAMERA_INDEX = 0
WINDOW_NAME = "Hand Filter Window  (q = quit, l = landmarks, h = clean view)"

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

SMOOTHING = 0.35          # 0..1, how fast the ROI follows the hands (lower = smoother)
MIN_ROI_SIZE = 20         # pixels; smaller rectangles are ignored
HAND_LOST_GRACE = 6       # frames to keep the ROI after a hand briefly disappears
PINCH_THRESHOLD = 0.35    # thumb-finger distance, as a fraction of hand size
GESTURE_HOLD_FRAMES = 4   # a pinch must be held this many frames to switch filters
COLOR_CYCLE_SECONDS = 8   # Color Scale: time for one full trip around the color wheel
COLOR_SCALE_SATURATION = 200  # Color Scale: 0 = grayscale, 255 = fully saturated tint
# Color Scale: colors stepped through by repeating the gesture, as (name, OpenCV hue 0..179).
COLOR_SCALE_PALETTE = [
    ("Red", 0),
    ("Orange", 12),
    ("Yellow", 28),
    ("Green", 60),
    ("Cyan", 90),
    ("Blue", 120),
    ("Purple", 140),
    ("Pink", 165),
]

# MediaPipe hand landmark indices (21 points per hand).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

# Lines between landmarks, used only for the optional skeleton overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                    # palm base
]

# Colors are BGR (OpenCV's channel order).
ROI_COLOR = (80, 255, 120)
LEFT_COLOR = (255, 160, 60)
RIGHT_COLOR = (60, 120, 255)
TEXT_COLOR = (255, 255, 255)


# --------------------------------------------------------------------------
# Filters: each takes a BGR image and returns a BGR image of the same size.
# To add a filter, write a function and add it to FILTERS / GESTURE_FILTERS.
# --------------------------------------------------------------------------

def grayscale(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class ColorScale:
    """
    Like grayscale, but in a single color: brightness comes from the image and
    every pixel shares one hue.

    Starts in "flow" mode, where the hue drifts around the color wheel. Each
    repeat of the gesture locks onto the next color in COLOR_SCALE_PALETTE;
    stepping past the last color goes back to flow.
    """

    def __init__(self):
        self.index = None  # None = flow, otherwise a position in COLOR_SCALE_PALETTE

    def reset(self):
        self.index = None

    def next_color(self):
        if self.index is None:
            self.index = 0
        elif self.index + 1 < len(COLOR_SCALE_PALETTE):
            self.index += 1
        else:
            self.index = None

    @property
    def label(self):
        if self.index is None:
            return "Color Scale: Flow"
        return f"Color Scale: {COLOR_SCALE_PALETTE[self.index][0]}"

    def __call__(self, roi):
        # OpenCV stores hue as 0..179.
        if self.index is None:
            hue = int(time.monotonic() / COLOR_CYCLE_SECONDS * 180) % 180
        else:
            hue = COLOR_SCALE_PALETTE[self.index][1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = np.empty_like(roi)
        hsv[..., 0] = hue
        hsv[..., 1] = COLOR_SCALE_SATURATION
        hsv[..., 2] = gray
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


color_scale = ColorScale()


def blur(roi):
    return cv2.GaussianBlur(roi, (41, 41), 0)


def pixelate(roi, block_size=16):
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


FILTERS = {
    "Grayscale": grayscale,
    "Color Scale": color_scale,
    "Blur": blur,
    "Pixelate": pixelate,
}

# Which fingertip, touched by the thumb, selects which filter.
GESTURE_FILTERS = {
    INDEX_TIP: "Grayscale",
    MIDDLE_TIP: "Color Scale",
    RING_TIP: "Blur",
    PINKY_TIP: "Pixelate",
}


# --------------------------------------------------------------------------
# Hand detection
# --------------------------------------------------------------------------

def ensure_model():
    """Download the hand landmark model once. Inference itself runs locally."""
    if not MODEL_PATH.exists():
        print(f"Downloading hand landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame, timestamp_ms):
    """
    Run the hand model on a BGR frame.

    Returns a dict like {"Left": [(x, y), ...21 points], "Right": [...]}
    with landmark positions in pixels. Missing hands are simply absent.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)

    hands = {}
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        points = [(lm.x * w, lm.y * h) for lm in landmarks]
        if label in hands:
            # Occasionally both hands get the same label; tell them apart by
            # which side of the image the wrist is on.
            other = hands.pop(label)
            pair = sorted([other, points], key=lambda p: p[WRIST][0])
            hands["Left"], hands["Right"] = pair
        else:
            hands[label] = points
    return hands


# --------------------------------------------------------------------------
# ROI: four fingertips -> rectangle
# --------------------------------------------------------------------------

def points_to_rect(points):
    """
    Turn any set of points into an axis-aligned rectangle.

    The points can arrive in any order: the left edge is the smallest x,
    the right edge the largest x, and the same for top/bottom with y.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def smooth_rect(previous, current, alpha=SMOOTHING):
    """Exponential moving average: move a fraction of the way toward the new rect."""
    if previous is None:
        return current
    return tuple(p + alpha * (c - p) for p, c in zip(previous, current))


def clamp_rect(rect, width, height):
    """Round to integer pixels and keep the rectangle inside the image."""
    x1, y1, x2, y2 = rect
    x1 = int(max(0, min(width, x1)))
    x2 = int(max(0, min(width, x2)))
    y1 = int(max(0, min(height, y1)))
    y2 = int(max(0, min(height, y2)))
    return x1, y1, x2, y2


# --------------------------------------------------------------------------
# Gestures: thumb-to-finger distance -> filter choice
# --------------------------------------------------------------------------

def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def detect_pinch(points):
    """
    Return (finger_tip_index, normalized_distance) for the finger closest to
    the thumb, or None if no finger is touching the thumb.

    Distances are divided by the hand's size (wrist -> middle knuckle), so the
    gesture works the same whether the hand is near or far from the camera.
    """
    hand_size = distance(points[WRIST], points[MIDDLE_MCP])
    if hand_size < 1:
        return None

    thumb = points[THUMB_TIP]
    best = None
    for tip in GESTURE_FILTERS:
        d = distance(thumb, points[tip]) / hand_size
        if d < PINCH_THRESHOLD and (best is None or d < best[1]):
            best = (tip, d)
    return best


class GestureSelector:
    """
    Remembers the current filter and only switches after the same pinch has
    been seen for several frames in a row, which stops flickering.

    A pinch fires once, when it has been held long enough; to fire again you
    let go and pinch again. After each update, `event` says what happened this
    frame: None, "selected" (switched to a new filter) or "repeated" (pinched
    the filter that was already active).
    """

    def __init__(self, initial):
        self.current = initial
        self.candidate = None
        self.count = 0
        self.event = None

    def update(self, hands):
        self.event = None
        pinches = [p for p in (detect_pinch(pts) for pts in hands.values()) if p]
        if not pinches:
            self.candidate, self.count = None, 0
            return self.current

        tip, _ = min(pinches, key=lambda p: p[1])  # tightest pinch from either hand
        wanted = GESTURE_FILTERS[tip]
        if wanted == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = wanted, 1

        if self.count == GESTURE_HOLD_FRAMES:
            self.event = "repeated" if wanted == self.current else "selected"
            self.current = wanted
        return self.current


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def draw_label(frame, text, origin, color=ROI_COLOR):
    """Text on a filled background box so it stays readable on any image."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(th + baseline + 4, y)  # keep the label on screen
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x + tw + 12, y), color, -1)
    cv2.putText(frame, text, (x + 6, y - baseline - 3), font, scale,
                (20, 20, 20), thickness, cv2.LINE_AA)


def draw_landmarks(frame, hands):
    for label, points in hands.items():
        color = LEFT_COLOR if label == "Left" else RIGHT_COLOR
        pts = [(int(x), int(y)) for x, y in points]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_tracked_points(frame, hands):
    """Highlight the four fingertips that define the ROI."""
    for label, points in hands.items():
        color = LEFT_COLOR if label == "Left" else RIGHT_COLOR
        for idx in (THUMB_TIP, INDEX_TIP):
            x, y = int(points[idx][0]), int(points[idx][1])
            cv2.circle(frame, (x, y), 9, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 9, TEXT_COLOR, 2, cv2.LINE_AA)


def draw_hud(frame, filter_name, hands_found, show_landmarks):
    h = frame.shape[0]
    status = "Both hands: framing" if hands_found == 2 else f"Show both hands ({hands_found}/2)"
    lines = [
        f"Filter: {filter_name}   |   {status}",
        "Pinch thumb + index/middle/ring/pinky to change filter",
        f"q = quit   l = landmarks ({'on' if show_landmarks else 'off'})   h = clean view",
    ]
    font, scale, line_height = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 24
    panel_w = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines) + 24
    panel_top = h - 10 - len(lines) * line_height

    # Darken a panel behind the text so it is readable on any background.
    panel = frame[panel_top:h, 0:panel_w]
    panel[:] = (panel * 0.45).astype(np.uint8)

    for i, line in enumerate(lines):
        y = panel_top + (i + 1) * line_height - 4
        cv2.putText(frame, line, (12, y), font, scale, TEXT_COLOR, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Per-frame pipeline
# --------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.rect = None          # smoothed ROI as floats (x1, y1, x2, y2), or None
        self.frames_missing = 0   # consecutive frames without both hands
        self.selector = GestureSelector(initial="Grayscale")
        self.show_landmarks = True
        self.clean_view = True    # True = only the filtered window, no overlays


def process_frame(frame, hands, state):
    """Update the ROI from the detected hands and draw everything onto frame."""
    h, w = frame.shape[:2]
    filter_name = state.selector.update(hands)
    display_name = filter_name

    if filter_name == "Color Scale":
        if state.selector.event == "selected":
            color_scale.reset()       # always arrive in flow mode
        elif state.selector.event == "repeated":
            color_scale.next_color()  # step to the next fixed color, then back to flow
        display_name = color_scale.label

    # 1. Update (or forget) the ROI.
    if "Left" in hands and "Right" in hands:
        corners = [hands[side][tip] for side in ("Left", "Right")
                   for tip in (THUMB_TIP, INDEX_TIP)]
        state.rect = smooth_rect(state.rect, points_to_rect(corners))
        state.frames_missing = 0
    else:
        state.frames_missing += 1
        if state.frames_missing > HAND_LOST_GRACE:
            state.rect = None  # hands gone: remove the window and restart smoothing

    # 2. Filter the ROI and paste it back into the frame.
    if state.rect is not None:
        x1, y1, x2, y2 = clamp_rect(state.rect, w, h)
        if x2 - x1 >= MIN_ROI_SIZE and y2 - y1 >= MIN_ROI_SIZE:
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = FILTERS[filter_name](roi)
            if not state.clean_view:
                cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 3, cv2.LINE_AA)
                draw_label(frame, display_name, (x1, y1 - 6))

    # 3. Overlays (all skipped in clean view).
    if state.clean_view:
        return frame
    if state.show_landmarks:
        draw_landmarks(frame, hands)
    draw_tracked_points(frame, hands)
    draw_hud(frame, display_name, len(hands), state.show_landmarks)
    return frame


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    ensure_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open the webcam. Check that it is connected and that your "
            "terminal has camera permission (macOS: System Settings > Privacy & "
            "Security > Camera)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = AppState()
    start = time.monotonic()
    last_timestamp = -1

    try:
        with create_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read a frame from the webcam; exiting.")
                    break

                # Mirror the image so it behaves like a mirror. MediaPipe's
                # Left/Right labels assume a mirrored (selfie) image.
                frame = cv2.flip(frame, 1)

                # VIDEO mode needs strictly increasing timestamps.
                timestamp = max(int((time.monotonic() - start) * 1000), last_timestamp + 1)
                last_timestamp = timestamp

                hands = detect_hands(landmarker, frame, timestamp)
                process_frame(frame, hands, state)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("l"):
                    state.show_landmarks = not state.show_landmarks
                if key == ord("h"):
                    state.clean_view = not state.clean_view
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # lets macOS actually close the window


if __name__ == "__main__":
    main()
