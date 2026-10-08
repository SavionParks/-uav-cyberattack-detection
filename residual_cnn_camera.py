"""
Residual CNN + repository-style real-time traffic-sign camera.

This keeps the same camera/detector pattern used by the GitHub repository:
- Ultralytics YOLO for locating signs
- OpenCV VideoCapture source "0" for the built-in webcam
- 1280x720 requested camera resolution
- 640 YOLO inference size by default
- Q to quit

The Residual CNN is used only as a second-stage classifier on each YOLO crop.
No adversarial perturbation is applied to the live camera feed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


# ------------------------------- SETTINGS --------------------------------

YOLO_WEIGHTS = ""  # leave empty to auto-find a .pt file
CLASSIFIER = "residual_cnn_traffic_signs.keras"
CLASS_NAMES_FILE = "residual_classes.json"

SOURCE = "0"       # "0" built-in webcam, "1" USB camera, or video path
CONF = 0.50
IMGSZ = 640
CNN_IMG_SIZE = 64

# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

COLORS = {
    "stop_sign": (0, 0, 255),
    "35_speed_limit": (0, 200, 0),
    "45_speed_limit": (0, 140, 255),
}

YOLO_TO_CNN = {
    "stop_sign": "Stop",
    "35_speed_limit": "speedLimit35",
    "45_speed_limit": "speedLimit45",
}


def resolve_file(path_text, extensions=None, auto_find=False):
    if path_text:
        p = Path(path_text)
        for candidate in (p, SCRIPT_DIR / p):
            if candidate.is_file():
                return candidate
        sys.exit(f"Could not find file: {path_text}")

    if not auto_find:
        return None

    candidates = []
    for folder in (SCRIPT_DIR, Path.home() / "Downloads"):
        if not folder.is_dir():
            continue
        if extensions:
            for ext in extensions:
                candidates.extend(folder.glob(f"*{ext}"))

    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        sys.exit("No compatible model file was found.")

    # Prefer a trained/custom best.pt over generic starter weights.
    custom = [
        p for p in candidates
        if not p.name.lower().startswith(("yolo", "yolov"))
    ]
    if custom:
        candidates = custom

    return max(candidates, key=lambda p: p.stat().st_mtime)


def open_source(source):
    if source.isdigit():
        index = int(source)

        if sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            cap = cv2.VideoCapture(index)

        if not cap.isOpened():
            sys.exit(
                f"Could not open camera {index}. "
                'Close Zoom/Teams/Camera or try --source "1".'
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    path = Path(source)
    if not path.is_file() and (SCRIPT_DIR / path).is_file():
        path = SCRIPT_DIR / path

    if not path.is_file():
        sys.exit(f"Could not find video: {source}")

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}")

    return cap


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)

    if not isinstance(names, list) or not names:
        raise ValueError("Class-name JSON must contain a non-empty list.")

    return names


def classify_crop(classifier, class_names, crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(
        crop_rgb,
        (CNN_IMG_SIZE, CNN_IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    x = crop_rgb.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    probs = classifier.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    return class_names[idx], conf


def main():
    ap = argparse.ArgumentParser(
        description="YOLO + Residual CNN traffic-sign camera"
    )

    ap.add_argument("--weights", default=YOLO_WEIGHTS)
    ap.add_argument("--classifier", default=CLASSIFIER)
    ap.add_argument("--classes", default=CLASS_NAMES_FILE)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)

    args, _ = ap.parse_known_args()

    yolo_weights = resolve_file(
        args.weights,
        extensions=[".pt"],
        auto_find=True
    )

    classifier_path = resolve_file(
        args.classifier,
        auto_find=False
    )

    class_names_path = resolve_file(
        args.classes,
        auto_find=False
    )

    print("Loading YOLO:", yolo_weights)
    detector = YOLO(str(yolo_weights))

    print("Loading Residual CNN:", classifier_path)
    classifier = tf.keras.models.load_model(
        str(classifier_path)
    )

    class_names = load_class_names(class_names_path)

    print("Residual CNN classes:", class_names)
    print("YOLO classes:", detector.names)

    cap = open_source(str(args.source))

    window = "Traffic Sign Detection + Residual CNN (Q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    fps = 0.0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Camera/video feed ended.")
            break

        result = detector.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False
        )[0]

        names = detector.names

        if result.boxes is not None:
            boxes = result.boxes.xyxy.int().tolist()
            classes = result.boxes.cls.int().tolist()
            scores = result.boxes.conf.tolist()

            for (x1, y1, x2, y2), cls_id, yolo_conf in zip(
                boxes,
                classes,
                scores
            ):
                yolo_name = names[cls_id]

                # Only send the three traffic-sign classes to the Residual CNN.
                if yolo_name not in YOLO_TO_CNN:
                    continue

                h, w = frame.shape[:2]

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                cnn_name, cnn_conf = classify_crop(
                    classifier,
                    class_names,
                    crop
                )

                expected_cnn_name = YOLO_TO_CNN[yolo_name]
                agrees = cnn_name == expected_cnn_name

                col = COLORS.get(yolo_name, (0, 255, 255))

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    col,
                    3
                )

                yolo_text = (
                    f"YOLO: {yolo_name} {yolo_conf:.2f}"
                )

                cnn_text = (
                    f"Residual CNN: {cnn_name} {cnn_conf:.2f}"
                )

                status_text = (
                    "AGREE" if agrees else "CHECK"
                )

                cv2.putText(
                    frame,
                    yolo_text,
                    (x1, max(25, y1 - 45)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    col,
                    2
                )

                cv2.putText(
                    frame,
                    cnn_text,
                    (x1, max(45, y1 - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    col,
                    2
                )

                cv2.putText(
                    frame,
                    status_text,
                    (x1, min(h - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    col,
                    2
                )

        now = time.time()
        fps = 0.9 * fps + 0.1 * (
            1.0 / max(now - t_prev, 1e-6)
        )
        t_prev = now

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 40),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"{fps:.1f} FPS",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

        if cv2.getWindowProperty(
            window,
            cv2.WND_PROP_VISIBLE
        ) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
