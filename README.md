# Hand Filter Window

Hold both hands up to your webcam and pinch the air into a live "filter window".
Your thumbs and index fingers set the corners of a rectangle. Everything inside
it gets a visual filter. Touch your thumb to a different finger to switch filters.

Everything runs locally on your machine: OpenCV for the camera and image
processing, and MediaPipe for hand tracking. No cloud APIs are used.

| Gesture (either hand)  | Filter      |
|------------------------|-------------|
| thumb touches index    | Grayscale   |
| thumb touches middle   | Color Scale |
| thumb touches ring     | Blur        |
| thumb touches pinky    | Pixelate    |

Keys: **`q`** quits, **`l`** turns the hand-skeleton overlay on or off, and
**`h`** toggles the clean view. The app starts in clean view, which shows only
the filtered window, with no border, labels, hand tracing or status bar. Press
`h` to bring them back, and again to hide them.

---

## How it works

### What is computer vision?

Computer vision is the field of getting computers to make sense of images and
video. To a program, a webcam frame is just a grid of numbers: a 1280×720 image
is 921,600 pixels, each with a blue, green and red value. Computer vision
algorithms turn that grid into useful facts, such as "there is a hand here and
its index fingertip is at pixel (640, 310)".

This app uses two layers of it:

- **Image processing** (OpenCV): operations directly on pixels, like converting
  to grayscale, blurring, or resizing.
- **Machine-learning perception** (MediaPipe): a neural network that looks at
  the frame and finds hands.

### What are hand landmarks?

A *landmark* is a specific, named point on an object. MediaPipe's hand model
returns 21 landmarks for each hand it sees, always in the same order:

```
        8   12  16  20        tips:     4 thumb, 8 index, 12 middle,
        |   |   |   |                   16 ring, 20 pinky
        7   11  15  19
        |   |   |   |
   4    6   10  14  18
    \   |   |   |   |
     3  5---9---13--17        knuckles: 5, 9, 13, 17
      \  \          /
       2  \        /
        \  \      /
         1--0----             0 = wrist
```

Each landmark is an (x, y) position, given as a fraction of the image size
(0.0–1.0). The app multiplies by the frame's width and height to get pixels.
MediaPipe also says whether each hand is the **Left** or the **Right** hand.

The app mirrors the camera image first, so the screen acts like a mirror. That
also matches what MediaPipe expects when it decides which hand is Left and
which is Right.

### What is an ROI (Region of Interest)?

An ROI is the part of an image you care about. Here, the ROI is the rectangle
between your hands. The app:

1. Cuts the ROI out of the frame (`frame[y1:y2, x1:x2]`). With NumPy this is
   only a slice, so nothing is copied.
2. Runs a filter on that small image.
3. Writes the filtered pixels back into the same spot in the frame.

The rest of the frame is left alone, which creates the "window" effect.

### Turning four fingertips into a rectangle

The four tracked points are the **left thumb tip**, **left index tip**,
**right thumb tip** and **right index tip**. They can come in any order. Your
hands might be crossed, tilted or upside down, so you can't assume which point
is "top-left".

The app doesn't need to know. An axis-aligned rectangle (edges parallel to the
screen) is fully set by four numbers, and each one comes from a simple min or
max over the points:

```python
left   = min(x of all four points)
right  = max(x of all four points)
top    = min(y of all four points)   # y grows downward in images
bottom = max(y of all four points)
```

This is the *bounding box* of the points, the smallest upright rectangle that
contains all four. It works no matter how the points are ordered.

A few more steps keep it stable and safe:

- **Smoothing.** Hand tracking jitters by a pixel or two from frame to frame.
  Each new rectangle is blended with the previous one, moving 35% of the way
  toward the new position (an *exponential moving average*). The window glides
  instead of shaking.
- **Clamping.** If your hands go past the edge of the frame, the coordinates
  are clamped to the image bounds, so slicing never fails.
- **Minimum size.** Rectangles smaller than 20 px are skipped.
- **Missing hands.** If a hand disappears, the window stays for a few frames
  (this covers brief tracking dropouts), then it's removed and the smoothing
  resets. When your hands come back, the window appears at their position
  without sliding in from the old one.

### Picking filters with gesture distances

A "touch" is detected by measuring the distance from the thumb tip to each of
the other four fingertips.

Raw pixel distances aren't reliable, though: a hand close to the camera looks
much bigger than one far away. So every distance is divided by the **hand's
size**, measured as the distance from the wrist (landmark 0) to the middle
knuckle (landmark 9):

```python
hand_size = distance(wrist, middle_knuckle)
pinch     = distance(thumb_tip, finger_tip) / hand_size
```

If that ratio is below `0.35`, the finger counts as touching the thumb. If more
than one finger is close, the closest one wins. No screen positions are
hardcoded. Only the relative geometry of the hand matters.

To stop the filter from flickering during a half-made gesture, a pinch has to
be held for 4 frames in a row before the filter changes. The chosen filter then
stays until you pinch a different finger.

### Filters

Each filter is a plain function that takes the ROI image and returns a new
image of the same size:

| Filter      | How it works |
|-------------|--------------|
| Grayscale   | Convert to one brightness channel, then back to 3 channels so it can be pasted into the color frame. |
| Color Scale | Like grayscale, but in one color: brightness from the image, a single hue for every pixel. Starts flowing around the color wheel (`COLOR_CYCLE_SECONDS`); pinch thumb + middle again to lock onto the next color in `COLOR_SCALE_PALETTE`, and after the last color it returns to the flow. |
| Blur        | Gaussian blur: each pixel becomes a weighted average of its neighbors. |
| Pixelate    | Shrink the ROI 16× and enlarge it again with nearest-neighbor scaling, which makes big blocks. |

To add a filter, write a function `my_filter(roi) -> image` in `app.py`, add it
to `FILTERS`, and map a fingertip to it in `GESTURE_FILTERS` (or replace an
existing mapping).

---

## Install and run

You need **Python 3.11 or 3.12** and a webcam.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

On first run, the app downloads MediaPipe's hand model
(`hand_landmarker.task`, about 8 MB) next to `app.py`. After that it works
offline.

**macOS:** the first time, macOS asks whether your terminal (Terminal, iTerm,
VS Code, …) can use the camera. If you said no, turn it back on under
*System Settings → Privacy & Security → Camera*.

### Troubleshooting

- **"Could not open the webcam".** Another app may be using the camera, or the
  terminal lacks camera permission. If you have several cameras, change
  `CAMERA_INDEX` at the top of `app.py`.
- **`Check failed: service_ Service is unavailable` on macOS.** You have
  MediaPipe 1.0.x installed, which crashes during setup on macOS. Install the
  pinned version: `pip install -r requirements.txt`.
- **Strange `cv2` import errors.** Make sure only one OpenCV package is
  installed: `pip uninstall opencv-python`, then reinstall the requirements.
  MediaPipe already brings in `opencv-contrib-python`.
- **The window feels laggy or too jumpy.** Adjust `SMOOTHING` in `app.py`
  (lower = smoother, higher = more responsive).

## Files

- `app.py`: the whole application, split into sections (config, filters,
  detection, ROI, gestures, drawing, main loop).
- `requirements.txt`: dependencies (just MediaPipe, which brings OpenCV and
  NumPy).
- `hand_landmarker.task`: the hand model, downloaded automatically.
