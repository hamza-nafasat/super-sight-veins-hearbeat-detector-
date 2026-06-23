"""
SuperSight — Milestone 1: Webcam Test
======================================
This script opens your laptop's default webcam and shows live video.
It is the first step toward measuring heart rate from facial color changes (rPPG).

Press 'q' to quit cleanly.
"""

import cv2


def main() -> None:
  # -------------------------------------------------------------------------
  # Step 1: Open the default webcam (device index 0).
  # On most laptops, index 0 is the built-in camera. If you have multiple
  # cameras, you might need index 1 or 2 instead.
  # -------------------------------------------------------------------------
  camera = cv2.VideoCapture(0)

  if not camera.isOpened():
    print("ERROR: Could not open webcam at index 0.")
    print("Check that your camera is connected and not in use by another app.")
    return

  # -------------------------------------------------------------------------
  # Step 2: Read the camera's actual resolution and frame rate.
  # These are *requested* values — the driver may round or override them.
  # We print them once at startup so you know what the camera reports.
  # -------------------------------------------------------------------------
  frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = camera.get(cv2.CAP_PROP_FPS)

  print("SuperSight M1 — Webcam opened successfully")
  print(f"  Frame width:  {frame_width} px")
  print(f"  Frame height: {frame_height} px")
  print(f"  FPS:          {fps}")
  print("Press 'q' in the video window to quit.")

  # Window title shown in the title bar of the live preview.
  window_title = "SuperSight M1"

  # -------------------------------------------------------------------------
  # Step 3: Main loop — grab frames and display them until the user quits.
  # -------------------------------------------------------------------------
  while True:
    # read() returns (success, frame). success is False if the frame failed.
    success, frame = camera.read()

    if not success:
      print("ERROR: Failed to read a frame from the webcam.")
      break

    # Show the current frame in a window. OpenCV handles the GUI on Linux
    # when you have a display (X11 or Wayland) available.
    cv2.imshow(window_title, frame)

    # waitKey(1) waits 1 ms for a keypress and returns its code.
    # We mask with 0xFF so we get a simple byte on all platforms.
    # If the user presses 'q', we break out of the loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
      print("Quit key pressed — shutting down.")
      break

  # -------------------------------------------------------------------------
  # Step 4: Clean up — release the camera and destroy all OpenCV windows.
  # Always do this so the camera is free for other apps.
  # -------------------------------------------------------------------------
  camera.release()
  cv2.destroyAllWindows()
  print("Camera released. Goodbye.")


if __name__ == "__main__":
  main()
