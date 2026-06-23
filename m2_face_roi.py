"""
SuperSight — Milestone 2: Face Detection + Forehead ROI
=========================================================
Builds on M1's webcam loop. Uses MediaPipe Face Landmarker to find your
face each frame, draws a light face outline, and places a green rectangle
on the forehead — the region we'll measure for pulse in later milestones.

Press 'q' to quit cleanly.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing
from mediapipe.tasks.python.vision import face_landmarker as mp_face_landmarker
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode

# ---------------------------------------------------------------------------
# MediaPipe landmark indices for the forehead ROI
# ---------------------------------------------------------------------------
# MediaPipe assigns each face mesh point a fixed index (478 points total).
# We pick landmarks on the eyebrows, forehead top, and temples to build a
# box on the forehead — above the eyebrows, between the temples.
#
# Person's LEFT eyebrow (appears on the RIGHT side of the webcam image):
LEFT_EYEBROW_INDICES = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
# Person's RIGHT eyebrow (appears on the LEFT side of the webcam image):
RIGHT_EYEBROW_INDICES = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
# Center of the forehead hairline (top of face oval):
FOREHEAD_TOP_INDEX = 10
# Temple points on the upper face oval — set the left/right width of the box:
LEFT_TEMPLE_INDEX = 109
RIGHT_TEMPLE_INDEX = 338

# Path to the bundled Face Landmarker model (downloaded in M2 setup).
MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"

# Colors (BGR format for OpenCV).
FOREHEAD_ROI_COLOR = (0, 255, 0)       # bright green — easy to spot
FACE_OUTLINE_COLOR = (180, 180, 180)   # light gray — subtle face outline
NO_FACE_TEXT_COLOR = (0, 0, 255)       # red warning text


def create_face_landmarker() -> FaceLandmarker:
  """
  Create a MediaPipe FaceLandmarker configured for live video.

  VIDEO mode lets MediaPipe track the face across frames (smoother and
  faster than treating every frame as a brand-new image).
  """
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
      num_faces=1,  # SuperSight tracks one person at a time.
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
  """
  Compute a forehead rectangle from face mesh landmarks.

  Returns (x1, y1, x2, y2) for cv2.rectangle — top-left and bottom-right.

  Strategy:
    - Bottom edge: just above the eyebrows (we don't want brow hair/motion).
    - Top edge: near landmark 10 (forehead center / hairline area).
    - Left/right edges: between the temples (landmarks 109 and 338).
  """
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

  # In image coordinates, smaller y = higher on screen.
  eyebrow_top_y = min(point[1] for point in eyebrow_points)

  # Rough vertical span from forehead top down to eyebrows — used for margins.
  face_span = max(eyebrow_top_y - forehead_top[1], 20)

  # Bottom of ROI: a little above the eyebrow line (exclude brow movement).
  bottom_y = eyebrow_top_y - int(face_span * 0.08)

  # Top of ROI: slightly above the forehead-center landmark.
  top_y = forehead_top[1] - int(face_span * 0.05)

  # Width: between temples, with a small inset so we stay on skin, not hair.
  temple_span = abs(right_temple[0] - left_temple[0])
  inset = int(temple_span * 0.10)
  left_x = min(left_temple[0], right_temple[0]) + inset
  right_x = max(left_temple[0], right_temple[0]) - inset

  # Guarantee a valid rectangle (at least a few pixels tall and wide).
  if bottom_y - top_y < 10:
    top_y = bottom_y - 10
  if right_x - left_x < 10:
    center_x = (left_x + right_x) // 2
    left_x = center_x - 5
    right_x = center_x + 5

  # Keep the box inside the frame.
  x1 = max(0, left_x)
  y1 = max(0, top_y)
  x2 = min(frame_width - 1, right_x)
  y2 = min(frame_height - 1, bottom_y)

  return x1, y1, x2, y2


def draw_face_outline(frame, landmarks) -> None:
  """Draw a light outline around the detected face using the face-oval mesh."""
  oval_connections = mp_face_landmarker.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
  outline_style = mp_drawing.DrawingSpec(
      color=FACE_OUTLINE_COLOR,
      thickness=1,
  )
  mp_drawing.draw_landmarks(
      image=frame,
      landmark_list=landmarks,
      connections=oval_connections,
      landmark_drawing_spec=None,       # don't draw individual dots
      connection_drawing_spec=outline_style,
      is_drawing_landmarks=False,
  )


def draw_no_face_message(frame) -> None:
  """Show a clear on-screen message when no face is detected."""
  text = "No face detected"
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.9
  thickness = 2

  # Center the text horizontally near the top of the frame.
  text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
  text_x = (frame.shape[1] - text_size[0]) // 2
  text_y = 40

  cv2.putText(
      frame,
      text,
      (text_x, text_y),
      font,
      font_scale,
      NO_FACE_TEXT_COLOR,
      thickness,
      cv2.LINE_AA,
  )


def main() -> None:
  # -------------------------------------------------------------------------
  # Step 1: Open the default webcam (same as M1).
  # -------------------------------------------------------------------------
  camera = cv2.VideoCapture(0)

  if not camera.isOpened():
    print("ERROR: Could not open webcam at index 0.")
    print("Check that your camera is connected and not in use by another app.")
    return

  frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = camera.get(cv2.CAP_PROP_FPS)

  print("SuperSight M2 — Webcam + Face Landmarker")
  print(f"  Frame width:  {frame_width} px")
  print(f"  Frame height: {frame_height} px")
  print(f"  FPS:          {fps}")
  print("Press 'q' in the video window to quit.")

  # -------------------------------------------------------------------------
  # Step 2: Load the Face Landmarker model (MediaPipe Tasks API).
  # -------------------------------------------------------------------------
  try:
    face_landmarker = create_face_landmarker()
  except FileNotFoundError as error:
    print(f"ERROR: {error}")
    camera.release()
    return

  window_title = "SuperSight M2"
  frame_timestamp_ms = 0

  # -------------------------------------------------------------------------
  # Step 3: Main loop — detect face, draw ROI, display frame.
  # -------------------------------------------------------------------------
  while True:
    success, frame = camera.read()

    if not success:
      print("ERROR: Failed to read a frame from the webcam.")
      break

    # MediaPipe expects RGB; OpenCV gives us BGR.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # VIDEO mode needs a monotonically increasing timestamp per frame.
    frame_timestamp_ms = int(time.time() * 1000)
    result = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if result.face_landmarks:
      # We asked for num_faces=1, so use the first (and only) detected face.
      landmarks = result.face_landmarks[0]

      draw_face_outline(frame, landmarks)

      x1, y1, x2, y2 = compute_forehead_roi(
          landmarks, frame_width, frame_height
      )
      cv2.rectangle(frame, (x1, y1), (x2, y2), FOREHEAD_ROI_COLOR, 2)
    else:
      # No face this frame — keep the video running, just show a message.
      draw_no_face_message(frame)

    cv2.imshow(window_title, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      print("Quit key pressed — shutting down.")
      break

  # -------------------------------------------------------------------------
  # Step 4: Clean up (same as M1).
  # -------------------------------------------------------------------------
  face_landmarker.close()
  camera.release()
  cv2.destroyAllWindows()
  print("Camera released. Goodbye.")


if __name__ == "__main__":
  main()
