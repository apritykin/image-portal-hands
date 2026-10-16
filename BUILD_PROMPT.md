# Build Prompt: Hand Filter Window

> Copy everything below the line into your coding agent (Claude Code, Cursor,
> Codex, etc.) from inside an empty folder. The agent should build the whole
> project from this description.

---

## Goal

Build **Hand Filter Window**: a local Python app that shows the webcam feed in
a window. When the user holds **both hands** up, their two thumb tips and two
index fingertips become the corners of a live rectangle, the "filter window".
Everything inside the rectangle gets a visual filter, and the rest of the frame
stays untouched. Touching the thumb to a different finger (either hand)
switches the filter:

| Gesture (either hand)  | Filter      |
|------------------------|-------------|
| thumb touches index    | Grayscale   |
| thumb touches middle   | Color Scale |
| thumb touches ring     | Blur        |
| thumb touches pinky    | Pixelate    |

It's a small demo of a real-time vision pipeline:

```
camera ─► hand landmarks ─► ROI from fingertips ─► pinch gesture ─► filter ROI ─► window
(OpenCV)   (MediaPipe         (bounding box +        (distance /       (OpenCV      (imshow)
           HandLandmarker)     smoothing + clamp)     hand size)        on a slice)
```

Everything runs locally in real time on a laptop CPU. The only network access
is a one-time download of the MediaPipe hand model (~8 MB). No cloud APIs.

## Constraints

- **Language:** Python 3.11 or 3.12
- **Dependencies:** put exactly this in `requirements.txt`, comments included:
  ```
  # MediaPipe also installs OpenCV (opencv-contrib-python) and NumPy.
  # Don't add opencv-python on top: two OpenCV packages conflict over `cv2`.
  # 1.0.x currently crashes on macOS (Metal "Service is unavailable"), so pin 0.10.35.
  mediapipe==0.10.35
  ```
  Use only the standard library beyond this (`math`, `time`,
  `urllib.request`, `pathlib`). Import `cv2`, `mediapipe as mp`, `numpy as np`,
  `BaseOptions` from `mediapipe.tasks.python`, and `vision` from
  `mediapipe.tasks.python`.
- **Files:** exactly these:
  ```
  hand-filter-window/
  ├── app.py                 # the whole app, ~490 lines
  ├── requirements.txt
  ├── README.md
  └── hand_landmarker.task   # downloaded at runtime next to app.py
  ```
- One script. Plain functions plus three small classes (`ColorScale`,
  `GestureSelector`, `AppState`). No frameworks, no CLI flags, no extra
  modules. Split `app.py` into sections with banner comments
  (`# ----` lines around a title): Configuration, Filters, Hand detection,
  ROI: four fingertips -> rectangle, Gestures: thumb-to-finger distance ->
  filter choice, Drawing, Per-frame pipeline, Main loop.
- Use the **MediaPipe Tasks API** (`mediapipe.tasks.python.vision`), not the
  legacy `mp.solutions` API.
- Use OpenCV for all image work. Colors are BGR.

## `app.py` specification

### Module docstring
Title `Hand Filter Window` underlined with `=`. Explain in two sentences that
thumbs and index fingertips set the corners of a live filter window, and a
thumb-to-finger touch picks the filter. List the four gesture → filter
mappings (note that thumb + middle again steps through fixed colors), then the
keys line: `q = quit   l = toggle landmark overlay   h = clean view (filter only)`.

### Configuration
- `CAMERA_INDEX = 0`
- `WINDOW_NAME = "Hand Filter Window  (q = quit, l = landmarks, h = clean view)"`
- `MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")`
- `MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"`
- Tuning constants, each with a short trailing comment:
  | Name | Value | Meaning |
  |------|-------|---------|
  | `SMOOTHING` | 0.35 | 0..1, how fast the ROI follows the hands (lower = smoother) |
  | `MIN_ROI_SIZE` | 20 | pixels; smaller rectangles are ignored |
  | `HAND_LOST_GRACE` | 6 | frames to keep the ROI after a hand briefly disappears |
  | `PINCH_THRESHOLD` | 0.35 | thumb–finger distance, as a fraction of hand size |
  | `GESTURE_HOLD_FRAMES` | 4 | a pinch must be held this many frames to switch filters |
  | `COLOR_CYCLE_SECONDS` | 8 | Color Scale: time for one full trip around the color wheel |
  | `COLOR_SCALE_SATURATION` | 200 | Color Scale: 0 = grayscale, 255 = fully saturated tint |
- `COLOR_SCALE_PALETTE`: list of `(name, OpenCV hue 0..179)`: Red 0, Orange 12,
  Yellow 28, Green 60, Cyan 90, Blue 120, Purple 140, Pink 165.
- Landmark indices: `WRIST=0, THUMB_TIP=4, INDEX_TIP=8, MIDDLE_MCP=9,
  MIDDLE_TIP=12, RING_TIP=16, PINKY_TIP=20`.
- `HAND_CONNECTIONS`: a hand-written list of 21 `(a, b)` pairs (thumb
  0-1-2-3-4, index 0-5-6-7-8, middle 5-9-10-11-12, ring 9-13-14-15-16, pinky
  13-17-18-19-20, palm base 0-17), commented per finger. Used only for the
  optional skeleton overlay.
- Colors (BGR): `ROI_COLOR = (80, 255, 120)`, `LEFT_COLOR = (255, 160, 60)`,
  `RIGHT_COLOR = (60, 120, 255)`, `TEXT_COLOR = (255, 255, 255)`.

### Filters
Each filter takes a BGR image and returns a BGR image of the same size. The
section comment says how to add one (write a function, add it to `FILTERS` /
`GESTURE_FILTERS`).

- `grayscale(roi)`: BGR→GRAY→BGR so it can be pasted back into a color frame.
- **`ColorScale`** class (callable): brightness from the image, one hue for
  every pixel. Docstring explains flow mode vs palette stepping.
  - `index = None` means "flow"; otherwise a position in `COLOR_SCALE_PALETTE`.
  - `reset()` → flow. `next_color()`: flow → 0 → 1 → … → last → back to flow.
  - `label` property: `"Color Scale: Flow"` or `"Color Scale: <Name>"`.
  - `__call__(roi)`: in flow mode
    `hue = int(time.monotonic() / COLOR_CYCLE_SECONDS * 180) % 180`, else the
    palette hue. Build an HSV image with `np.empty_like(roi)`: H = hue,
    S = `COLOR_SCALE_SATURATION`, V = grayscale of the ROI; convert HSV→BGR.
  - A module-level instance `color_scale = ColorScale()`.
- `blur(roi)`: `cv2.GaussianBlur(roi, (41, 41), 0)`.
- `pixelate(roi, block_size=16)`: shrink to `max(1, w // block)` ×
  `max(1, h // block)` with `INTER_LINEAR`, enlarge back with `INTER_NEAREST`.
- `FILTERS` dict: `"Grayscale"`, `"Color Scale"` (the `color_scale` instance),
  `"Blur"`, `"Pixelate"`.
- `GESTURE_FILTERS` dict: `INDEX_TIP → "Grayscale"`, `MIDDLE_TIP → "Color
  Scale"`, `RING_TIP → "Blur"`, `PINKY_TIP → "Pixelate"`.

### Hand detection
- `ensure_model()`: if `MODEL_PATH` doesn't exist, print
  `Downloading hand landmark model to <path> ...` and fetch it with
  `urllib.request.urlretrieve`. Docstring: inference itself runs locally.
- `create_landmarker()`: `HandLandmarker` in `RunningMode.VIDEO`, `num_hands=2`,
  detection confidence 0.6, presence 0.5, tracking 0.5.
- `detect_hands(landmarker, frame, timestamp_ms)`: BGR→RGB into an
  `mp.Image(SRGB)`, run `detect_for_video`, and return a dict
  `{"Left": [(x, y) × 21], "Right": [...]}` in **pixels** (landmark x/y times
  frame width/height). Missing hands are simply absent. If both hands get the
  same handedness label, pop the existing one and assign Left/Right by sorting
  the two hands on wrist x.

### ROI: four fingertips → rectangle
- `points_to_rect(points)`: bounding box `(min x, min y, max x, max y)`. The
  docstring explains the points can arrive in any order.
- `smooth_rect(previous, current, alpha=SMOOTHING)`: exponential moving
  average per coordinate; return `current` if `previous` is `None`.
- `clamp_rect(rect, width, height)`: round to `int` and clamp each coordinate
  into the image.

### Gestures: thumb-to-finger distance → filter choice
- `distance(a, b)`: `math.hypot`.
- `detect_pinch(points)`: `hand_size = distance(WRIST, MIDDLE_MCP)`; return
  `None` if it's under 1 px. For each tip in `GESTURE_FILTERS`, compute
  `distance(THUMB_TIP, tip) / hand_size`. Return `(tip, d)` for the closest
  tip under `PINCH_THRESHOLD`, or `None`. Docstring: dividing by hand size
  makes it work near or far from the camera.
- **`GestureSelector`**: debounces the filter choice.
  - `__init__(initial)`: `current`, `candidate = None`, `count = 0`,
    `event = None`.
  - `update(hands)`: reset `event` to `None`. Collect pinches from all hands;
    if none, clear candidate/count and return `current`. Otherwise take the
    tightest pinch, map its tip to a filter name, increment `count` if it
    matches `candidate` (else start a new candidate at 1). When `count`
    reaches **exactly** `GESTURE_HOLD_FRAMES`, set `event` to `"repeated"` if
    it's the current filter or `"selected"` if not, and make it current. A
    pinch fires once; you have to let go and pinch again to fire again.
    Return `current`.

### Drawing
- `draw_label(frame, text, origin, color=ROI_COLOR)`: filled colored box with
  dark text `(20, 20, 20)` inside, `FONT_HERSHEY_SIMPLEX` scale 0.6, thickness
  2. Push `y` down so the label stays on screen.
- `draw_landmarks(frame, hands)`: 1-px skeleton lines from `HAND_CONNECTIONS`
  and 2-px dots, `LEFT_COLOR` for the left hand, `RIGHT_COLOR` for the right.
- `draw_tracked_points(frame, hands)`: highlight `THUMB_TIP` and `INDEX_TIP`
  on each hand with a filled 9-px circle in the hand color plus a white
  2-px outline.
- `draw_hud(frame, filter_name, hands_found, show_landmarks)`: three lines in a
  bottom-left panel:
  1. `Filter: <name>   |   Both hands: framing` (or `Show both hands (N/2)`)
  2. `Pinch thumb + index/middle/ring/pinky to change filter`
  3. `q = quit   l = landmarks (on|off)   h = clean view`

  Size the panel to the widest line + 24 px, darken it to 45% brightness in
  place (`panel[:] = (panel * 0.45).astype(np.uint8)`), then draw white text
  (scale 0.55, line height 24).

### Per-frame pipeline
- **`AppState`**: `rect = None` (smoothed float ROI), `frames_missing = 0`,
  `selector = GestureSelector(initial="Grayscale")`, `show_landmarks = True`,
  `clean_view = True` (the app **starts** in clean view).
- `process_frame(frame, hands, state)`:
  1. `filter_name = state.selector.update(hands)`. If it's `"Color Scale"`:
     on `"selected"` call `color_scale.reset()` (always arrive in flow mode),
     on `"repeated"` call `color_scale.next_color()`. The display name is
     `color_scale.label`.
  2. If both Left and Right are present, take the four corners (thumb + index
     tip of each hand), smooth the bounding box into `state.rect`, and reset
     `frames_missing`. Otherwise increment it, and once it passes
     `HAND_LOST_GRACE` set `rect = None` (so the window reappears at the hands
     instead of sliding in from the old spot).
  3. If there's a rect, clamp it. If both sides are ≥ `MIN_ROI_SIZE`, slice
     the ROI, run `FILTERS[filter_name]` on it, and write the result back into
     the same slice. Outside clean view, draw a 3-px `ROI_COLOR` border and a
     label with the display name just above it.
  4. In clean view, return now. Otherwise draw landmarks (if enabled), the
     tracked fingertips, and the HUD.

### Main loop
`main()`:
1. `ensure_model()`.
2. Open `cv2.VideoCapture(CAMERA_INDEX)`. If it fails, `raise SystemExit`
   with a message about checking the connection and camera permission
   (*macOS: System Settings > Privacy & Security > Camera*). Request
   1280×720.
3. Inside `with create_landmarker() as landmarker:` loop:
   1. Read a frame; on failure print a message and break.
   2. Mirror with `cv2.flip(frame, 1)`. Comment why: it behaves like a mirror,
      and MediaPipe's Left/Right labels assume a selfie image.
   3. Timestamp in ms, forced **strictly increasing**
      (`max(int(elapsed * 1000), last_timestamp + 1)`), because VIDEO mode
      requires it.
   4. `detect_hands`, `process_frame`, `imshow`.
   5. Keys via `cv2.waitKey(1) & 0xFF`: `q` quits, `l` toggles landmarks, `h`
      toggles clean view.
4. In `finally`, release the capture, `destroyAllWindows()`, then
   `cv2.waitKey(1)` (comment: lets macOS actually close the window).

## `README.md` specification

Write it for someone curious about computer vision, not just someone who wants
to run it. Plain, friendly language.

1. **Title and intro:** `# Hand Filter Window`. Two short paragraphs: the
   pinch-a-window trick, and that it runs locally with OpenCV + MediaPipe, no
   cloud APIs. Then the gesture → filter table.
2. **Keys:** `q` quits, `l` toggles the skeleton overlay, `h` toggles clean
   view. Explain that the app starts in clean view (only the filtered window:
   no border, labels, tracing, or status bar).
3. `---`, then **How it works**, with these subsections:
   - **What is computer vision?** A frame is a grid of numbers (1280×720 =
     921,600 pixels, each B/G/R). Two layers: image processing (OpenCV) and
     machine-learning perception (MediaPipe).
   - **What are hand landmarks?** 21 named points in a fixed order, with an
     ASCII hand diagram labeling tips (4, 8, 12, 16, 20), knuckles (5, 9, 13,
     17), and the wrist (0). Coordinates are fractions of the image size,
     converted to pixels. Left/Right handedness, and why the image is mirrored.
   - **What is an ROI (Region of Interest)?** Cut (`frame[y1:y2, x1:x2]`, a
     NumPy slice, no copy), filter, write back.
   - **Turning four fingertips into a rectangle:** points come in any order,
     the min/max bounding-box code block, then bullets for smoothing (35% EMA),
     clamping, minimum size (20 px), and missing hands (grace frames, then
     reset so it doesn't slide in).
   - **Picking filters with gesture distances:** why raw pixel distances fail,
     the `hand_size` / `pinch` code block, the 0.35 threshold, closest wins,
     held 4 frames, sticks until you pinch another finger.
   - **Filters:** a table explaining Grayscale, Color Scale (flow mode,
     `COLOR_CYCLE_SECONDS`, stepping through `COLOR_SCALE_PALETTE`, back to
     flow after the last color), Blur, Pixelate (16× shrink + nearest-neighbor
     enlarge). Then how to add your own filter.
4. `---`, then **Install and run:** Python 3.11 or 3.12 and a webcam; venv
   commands (with the Windows activate variant), `pip install -r
   requirements.txt`, `python app.py`. First run downloads
   `hand_landmarker.task` (~8 MB) next to `app.py`, then it works offline.
   macOS camera-permission note.
5. **Troubleshooting** bullets: "Could not open the webcam" (another app,
   permission, change `CAMERA_INDEX`); the MediaPipe 1.0.x
   `Check failed: service_ Service is unavailable` crash on macOS (reinstall
   the pinned requirements); strange `cv2` import errors (uninstall
   `opencv-python`); laggy or jumpy window (tune `SMOOTHING`).
6. **Files:** one line each for `app.py` (and its sections),
   `requirements.txt`, and `hand_landmarker.task`.

## Done when

- [ ] `pip install -r requirements.txt` works in a fresh Python 3.11/3.12 venv
- [ ] `python -c "import app"` succeeds with no camera (nothing runs on import)
- [ ] First run downloads `hand_landmarker.task` next to `app.py`; later runs
      skip it
- [ ] `python app.py` opens the webcam in clean view. With both hands up, a
      filtered rectangle follows the thumbs and index fingers, survives a brief
      tracking dropout, and disappears a few frames after a hand leaves.
      Each thumb-to-finger pinch switches to its filter. Repeating thumb +
      middle steps Flow → Red → … → Pink → Flow. `h` shows the border, label,
      fingertip markers, and HUD; `l` toggles the skeleton; `q` quits and the
      window closes. *(This is a manual check that needs a real camera. If
      you're an agent without one, say so instead of claiming it passed.)*
