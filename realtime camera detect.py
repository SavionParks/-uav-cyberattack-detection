"""
Real-time traffic sign detection (stop sign, 35 mph, 45 mph) with a YOLO model.

Setup (one time, in Command Prompt):
    py -m pip install ultralytics opencv-python

Easiest way to run:
    Put this file in the same folder as your model (.pt) file, open it in IDLE,
    and press F5. It finds the model automatically and opens your webcam.

To change settings, edit the SETTINGS section below. For example, set
SOURCE = "drive.mp4" to run on a video file instead of the webcam.

Close the window or press Q to quit.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ------------------------------- SETTINGS --------------------------------
WEIGHTS = ""    # leave empty to auto-find a .pt file, or put a full path here
SOURCE = "0"    # "0" = built-in webcam, "1" = USB camera, or a video like "drive.mp4"
CONF = 0.5      # lower (e.g. 0.3) to catch more signs, higher for fewer false alarms
IMGSZ = 640     # try 416 or 320 if the FPS is low on your laptop
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# OpenCV colors are in BGR order (blue, green, red).
COLORS = {
    "stop_sign": (0, 0, 255),         # red
    "35_speed_limit": (0, 200, 0),    # green
    "45_speed_limit": (0, 140, 255),  # orange
}
ALERTS = [
    ("stop_sign", "STOP SIGN AHEAD"),
    ("35_speed_limit", "SPEED LIMIT 35"),
    ("45_speed_limit", "SPEED LIMIT 45"),
]


def find_weights(choice):
    """Return the model file to use, searching sensible places if needed."""
    if choice:
        p = Path(choice)
        for candidate in (p, SCRIPT_DIR / p):
            if candidate.is_file():
                return candidate
        sys.exit(f"Could not find the model file: {choice}\n"
                 f"Check the name and path in the WEIGHTS setting.")

    search_dirs = [SCRIPT_DIR, Path.home() / "Downloads"]
    found = []
    for d in search_dirs:
        if d.is_dir():
            found += [f for f in d.glob("*.pt") if f.is_file()]
    # Skip the generic starter models Ultralytics may download (yolo11n.pt, etc.).
    found = [f for f in found if not f.name.lower().startswith(("yolo", "yolov"))]
    if not found:
        sys.exit("No model (.pt) file found.\n"
                 f"Put your model file in {SCRIPT_DIR}\n"
                 "or type its full path into the WEIGHTS setting at the top of this script.")

    def score(f):
        name = f.name.lower()
        return ("best" in name or "stop_sign" in name, f.stat().st_mtime)

    return max(found, key=score)


def open_source(source):
    """Open a webcam (by number) or a video file."""
    if source.isdigit():
        index = int(source)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            sys.exit(f"Could not open camera {index}.\n"
                     "Close other apps using the webcam (Zoom, Teams, Camera), "
                     "or try SOURCE = \"1\".")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    path = Path(source)
    if not path.is_file() and (SCRIPT_DIR / path).is_file():
        path = SCRIPT_DIR / path
    if not path.is_file():
        sys.exit(f"Could not find the video file: {source}\n"
                 f"Put it in {SCRIPT_DIR} or use its full path.")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Could not open the video file: {path}")
    return cap


def main():
    ap = argparse.ArgumentParser(description="Real-time traffic sign detection")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    args, _ = ap.parse_known_args()

    weights = find_weights(args.weights)
    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    names = model.names
    print(f"Classes: {list(names.values())}")

    cap = open_source(str(args.source))
    window = "Traffic Sign Detection (Q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Running... press Q in the video window (or close it) to stop.")

    fps, t_prev = 0.0, time.time()
    frames, t_start = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if not str(args.source).isdigit():
                print("Reached the end of the video.")
            else:
                print("Lost the camera feed.")
            break

        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]

        seen = set()
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.int().tolist(),
                                          r.boxes.cls.int().tolist(),
                                          r.boxes.conf.tolist()):
            name = names[c]
            seen.add(name)
            col = COLORS.get(name, (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
            cv2.putText(frame, f"{name} {s:.2f}", (x1, max(60, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6))
        t_prev = now
        frames += 1

        alert = next(((text, COLORS[cls]) for cls, text in ALERTS if cls in seen), None)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if alert:
            cv2.putText(frame, alert[0], (180, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert[1], 2)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    if frames and elapsed > 0:
        print(f"Processed {frames} frames, average {frames / elapsed:.1f} FPS.")


if __name__ == "__main__":
    main()
