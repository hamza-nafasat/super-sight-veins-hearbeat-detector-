"""
SuperSight — Milestone 3: Raw Color Signal from Forehead ROI
=============================================================
Builds on M2's face tracking and forehead box. Each frame we average the
pixel colors inside that box and store the green channel over time — the
raw input for pulse detection in later milestones.

We do NOT calculate heart rate here. M3 only proves we capture a steady
stream of color samples at a stable sampling rate.

Press 'q' to quit cleanly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode

# ---------------------------------------------------------------------------
# MediaPipe landmark indices for the forehead ROI (same as M2)
# ---------------------------------------------------------------------------
LEFT_EYEBROW_INDICES = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW_INDICES = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
FOREHEAD_TOP_INDEX = 10
LEFT_TEMPLE_INDEX = 109
RIGHT_TEMPLE_INDEX = 338

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

# How many seconds of signal history to keep in the rolling buffer.
BUFFER_DURATION_SEC = 10.0

# Colors (BGR for OpenCV).
FOREHEAD_ROI_COLOR = (0, 255, 0)
FACE_OUTLINE_COLOR = (180, 180, 180)
NO_FACE_TEXT_COLOR = (0, 0, 255)
HUD_TEXT_COLOR = (255, 255, 255)       # white overlay text
HUD_BACKGROUND_COLOR = (0, 0, 0)     # black panel behind text


@dataclass
class ColorSample:
  """One color measurement taken from the forehead ROI at a point in time."""

  timestamp_sec: float
  blue: float    # B channel average (OpenCV stores images as BGR)
  green: float   # G channel — our main pulse signal for rPPG
  red: float     # R channel average


class SignalBuffer:
  """
  Rolling buffer that keeps roughly the last BUFFER_DURATION_SEC seconds
  of color samples. Old samples fall off automatically.
  """

  def __init__(self, duration_sec: float) -> None:
    self.duration_sec = duration_sec
    self.samples: deque[ColorSample] = deque()

  def add(self, sample: ColorSample) -> None:
    """Add a new sample and remove anything older than the window."""
    self.samples.append(sample)
    self._prune_old_samples(sample.timestamp_sec)

  def _prune_old_samples(self, current_time_sec: float) -> None:
    cutoff = current_time_sec - self.duration_sec
    while self.samples and self.samples[0].timestamp_sec < cutoff:
      self.samples.popleft()

  def count(self) -> int:
    return len(self.samples)

  def measured_sampling_rate_hz(self) -> float:
    """
    Compute actual sampling rate (Hz) from stored timestamps.

    Uses (N - 1) intervals divided by total elapsed time — more accurate
    than assuming the webcam's reported FPS.
    """
    if len(self.samples) < 2:
      return 0.0

    start_time = self.samples[0].timestamp_sec
    end_time = self.samples[-1].timestamp_sec
    elapsed = end_time - start_time

    if elapsed <= 0.0:
      return 0.0

    return (len(self.samples) - 1) / elapsed

  def latest_green(self) -> float | None:
    if not self.samples:
      return None
    return self.samples[-1].green


def create_face_landmarker() -> FaceLandmarker:
  """Create a MediaPipe FaceLandmarker configured for live video."""
  if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Face Landmarker model not found at:\n  {MODEL_PATH}\n"
        "Download it with:\n"
        "  curl -o models/face_landmarker.task "
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

  options = FaceLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
      running_mode=running_mode.VisionTaskRunningMode.VIDEO,
      num_faces=1,
      min_face_detection_confidence=0.5,
      min_face_presence_confidence=0.5,
      min_tracking_confidence=0.5,
  )
  return FaceLandmarker.create_from_options(options)


def landmark_to_pixel(
    landmark,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
  """Convert a normalized landmark (0.0–1.0) to pixel (x, y) coordinates."""
  x_px = int(landmark.x * frame_width)
  y_px = int(landmark.y * frame_height)
  return x_px, y_px


def compute_forehead_roi(
    landmarks,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
  """Compute forehead rectangle (x1, y1, x2, y2) from face mesh landmarks."""
  eyebrow_indices = LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES
  eyebrow_points = [
      landmark_to_pixel(landmarks[i], frame_width, frame_height)
      for i in eyebrow_indices
  ]

  forehead_top = landmark_to_pixel(
      landmarks[FOREHEAD_TOP_INDEX], frame_width, frame_height
  )
  left_temple = landmark_to_pixel(
      landmarks[LEFT_TEMPLE_INDEX], frame_width, frame_height
  )
  right_temple = landmark_to_pixel(
      landmarks[RIGHT_TEMPLE_INDEX], frame_width, frame_height
  )

  eyebrow_top_y = min(point[1] for point in eyebrow_points)
  face_span = max(eyebrow_top_y - forehead_top[1], 20)

  bottom_y = eyebrow_top_y - int(face_span * 0.08)
  top_y = forehead_top[1] - int(face_span * 0.05)

  temple_span = abs(right_temple[0] - left_temple[0])
  inset = int(temple_span * 0.10)
  left_x = min(left_temple[0], right_temple[0]) + inset
  right_x = max(left_temple[0], right_temple[0]) - inset

  if bottom_y - top_y < 10:
    top_y = bottom_y - 10
  if right_x - left_x < 10:
    center_x = (left_x + right_x) // 2
    left_x = center_x - 5
    right_x = center_x + 5

  x1 = max(0, left_x)
  y1 = max(0, top_y)
  x2 = min(frame_width - 1, right_x)
  y2 = min(frame_height - 1, bottom_y)

  return x1, y1, x2, y2


def average_roi_color(
    frame,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[float, float, float] | None:
  """
  Compute the mean B, G, R values of all pixels inside the ROI.

  OpenCV images are BGR (not RGB). We return all three channels but
  later milestones will focus on GREEN because hemoglobin in blood
  absorbs green light more strongly than red or blue. When your heart
  pushes a pulse of blood into the face, slightly more green is absorbed,
  so the reflected green intensity dips in sync with your heartbeat.
  """
  # Slice the rectangular forehead region from the full frame.
  roi = frame[y1:y2, x1:x2]

  if roi.size == 0:
    return None

  # cv2.mean returns (B, G, R, alpha). We ignore alpha.
  mean_blue, mean_green, mean_red, _ = cv2.mean(roi)
  return mean_blue, mean_green, mean_red


def draw_face_outline(frame, landmarks) -> None:
  """Draw a light outline around the detected face."""
  oval_connections = mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
  outline_style = mp_drawing.DrawingSpec(
      color=FACE_OUTLINE_COLOR,
      thickness=1,
  )
  mp_drawing.draw_landmarks(
      image=frame,
      landmark_list=landmarks,
      connections=oval_connections,
      landmark_drawing_spec=None,
      connection_drawing_spec=outline_style,
      is_drawing_landmarks=False,
  )


def draw_hud(
    frame,
    *,
    green_value: float | None,
    sample_count: int,
    sampling_rate_hz: float,
    signal_paused: bool,
) -> None:
  """
  Draw a small heads-up display (HUD) with live signal stats.

  This is our proof that sampling works before we plot or filter anything.
  """
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.55
  thickness = 1
  line_height = 22
  padding = 8

  if signal_paused:
    lines = [
        "No face — signal paused",
        f"Samples in buffer: {sample_count}",
        f"Sampling rate: {sampling_rate_hz:.1f} Hz",
    ]
  else:
    green_text = f"{green_value:.2f}" if green_value is not None else "—"
    lines = [
        f"Green avg (raw): {green_text}",
        f"Samples in buffer: {sample_count}",
        f"Sampling rate: {sampling_rate_hz:.1f} Hz",
    ]

  # Measure the panel size so we can draw a readable background.
  max_width = 0
  for line in lines:
    text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
    max_width = max(max_width, text_size[0])

  panel_width = max_width + padding * 2
  panel_height = line_height * len(lines) + padding * 2

  # Semi-opaque black rectangle in the top-left corner.
  overlay = frame.copy()
  cv2.rectangle(
      overlay,
      (10, 10),
      (10 + panel_width, 10 + panel_height),
      HUD_BACKGROUND_COLOR,
      -1,
  )
  cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

  # Draw each line of text.
  y = 10 + padding + line_height - 6
  for line in lines:
    color = NO_FACE_TEXT_COLOR if signal_paused and line.startswith("No face") else HUD_TEXT_COLOR
    cv2.putText(
        frame,
        line,
        (10 + padding, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    y += line_height


def main() -> None:
  # -------------------------------------------------------------------------
  # Step 1: Open the default webcam (same as M1 / M2).
  # -------------------------------------------------------------------------
  camera = cv2.VideoCapture(0)

  if not camera.isOpened():
    print("ERROR: Could not open webcam at index 0.")
    print("Check that your camera is connected and not in use by another app.")
    return

  frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = camera.get(cv2.CAP_PROP_FPS)

  print("SuperSight M3 — Forehead color signal capture")
  print(f"  Frame width:  {frame_width} px")
  print(f"  Frame height: {frame_height} px")
  print(f"  FPS:          {fps}")
  print(f"  Buffer:       last {BUFFER_DURATION_SEC:.0f} seconds of samples")
  print("Press 'q' in the video window to quit.")

  # -------------------------------------------------------------------------
  # Step 2: Load face landmarker + create the rolling signal buffer.
  # -------------------------------------------------------------------------
  try:
    face_landmarker = create_face_landmarker()
  except FileNotFoundError as error:
    print(f"ERROR: {error}")
    camera.release()
    return

  signal_buffer = SignalBuffer(duration_sec=BUFFER_DURATION_SEC)
  window_title = "SuperSight M3"

  # -------------------------------------------------------------------------
  # Step 3: Main loop — detect face, sample ROI color, update buffer, display.
  # -------------------------------------------------------------------------
  while True:
    success, frame = camera.read()

    if not success:
      print("ERROR: Failed to read a frame from the webcam.")
      break

    sample_time_sec = time.perf_counter()
    signal_paused = True
    latest_green: float | None = None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    frame_timestamp_ms = int(time.time() * 1000)
    result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if result.face_landmarks:
      landmarks = result.face_landmarks[0]

      draw_face_outline(frame, landmarks)

      x1, y1, x2, y2 = compute_forehead_roi(
          landmarks, frame_width, frame_height
      )
      cv2.rectangle(frame, (x1, y1), (x2, y2), FOREHEAD_ROI_COLOR, 2)

      color_means = average_roi_color(frame, x1, y1, x2, y2)

      if color_means is not None:
        mean_blue, mean_green, mean_red = color_means

        # Store this frame's sample with a high-precision timestamp.
        signal_buffer.add(
            ColorSample(
                timestamp_sec=sample_time_sec,
                blue=mean_blue,
                green=mean_green,
                red=mean_red,
            )
        )
        latest_green = mean_green
        signal_paused = False

    draw_hud(
        frame,
        green_value=latest_green if latest_green is not None else signal_buffer.latest_green(),
        sample_count=signal_buffer.count(),
        sampling_rate_hz=signal_buffer.measured_sampling_rate_hz(),
        signal_paused=signal_paused,
    )

    cv2.imshow(window_title, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      print("Quit key pressed — shutting down.")
      break

  # -------------------------------------------------------------------------
  # Step 4: Clean up.
  # -------------------------------------------------------------------------
  face_landmarker.close()
  camera.release()
  cv2.destroyAllWindows()
  print("Camera released. Goodbye.")


if __name__ == "__main__":
  main()
