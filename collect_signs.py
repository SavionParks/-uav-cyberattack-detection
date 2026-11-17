"""
collect_signs.py - take labeled training pictures of YOUR signs with your webcam.

Your old model ("Detecting stop signs traffic (3).pt") already recognizes your phone signs,
so it draws the box for you. You only tell the tool WHICH sign you are holding up.
The saved pictures are CLEAN (no boxes or text drawn on them).

Run (in New folder (4)):
    py collect_signs.py

Keys (click the video window first):
    1       you are holding the 35 mph sign
    2       you are holding the 45 mph sign
    3       you are holding the STOP sign
    B       background mode: hold up NO sign (just you / the room)  -> teaches "not a sign"
    SPACE   start / pause saving
    Q       quit

How to use it:
    Press 2, hold up the 45, press SPACE. Move the phone around slowly: closer, farther,
    left, right, a little tilted, phone brighter / darker. Stop at about 150 pictures.
    Do the same for 1 (35) and 3 (stop). Then B for about 60 background pictures.

Output: a folder "my_signs" next to this script (images/ + labels/ + data.yaml).
Zip it and upload it to Google Drive, then run the training notebook.
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# ------------------------------- SETTINGS --------------------------------
LABEL_MODEL = "Detecting stop signs traffic (3).pt"   # the model that draws the boxes for you
SOURCE = "0"            # webcam number
MIN_CONF = 0.35         # the label model must be at least this sure about the box
SAVE_EVERY = 0.4        # seconds between saved pictures (so they aren't all the same)
GOAL_PER_SIGN = 150
GOAL_BACKGROUND = 60
OUT_DIR = "my_signs"
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
NAMES = ["35_speed_limit", "45_speed_limit", "stop_sign"]      # same order as your models
MODES = {ord("1"): 0, ord("2"): 1, ord("3"): 2, ord("b"): "bg", ord("B"): "bg"}
MODE_TEXT = {0: "35 mph", 1: "45 mph", 2: "STOP", "bg": "BACKGROUND (no sign)", None: "press 1 / 2 / 3 / B"}
COLORS = {0: (0, 200, 0), 1: (0, 140, 255), 2: (0, 0, 255)}


def find(path_text):
    for p in (Path(path_text), SCRIPT_DIR / path_text):
        if p.is_file():
            return p
    sys.exit(f"Could not find {path_text}. Put it in {SCRIPT_DIR} or change LABEL_MODEL.")


def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None] if sys.platform == "win32" else [None]
    for backend in backends:
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            continue
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None and frame.mean() > 8:
                return cap
        cap.release()
    sys.exit("The camera isn't giving a picture. Close other apps using it, or try --source 1.")


def main():
    ap = argparse.ArgumentParser(description="Collect labeled sign pictures from the webcam")
    ap.add_argument("--model", default=LABEL_MODEL)
    ap.add_argument("--source", default=SOURCE)
    args, _ = ap.parse_known_args()

    labeler = YOLO(str(find(args.model)))
    print("Labeling with:", args.model, "| its classes:", list(labeler.names.values()))

    out = SCRIPT_DIR / OUT_DIR
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        "# made by collect_signs.py\nnames:\n" + "".join(f"  - {n}\n" for n in NAMES))

    # count what is already there, so you can stop and continue later
    counts = {0: 0, 1: 0, 2: 0, "bg": 0}
    for lab in (out / "labels").glob("*.txt"):
        t = lab.read_text().strip()
        counts["bg" if not t else int(t.split()[0])] += 1

    cap = open_camera(int(args.source))
    window = "Collect sign pictures (1=35  2=45  3=STOP  B=background  SPACE=save on/off  Q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    mode, saving, last_save = None, False, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera feed.")
            break
        h, w = frame.shape[:2]
        r = labeler.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
        dets = [(b, float(s)) for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        best = max(dets, key=lambda d: d[1]) if dets else None

        # what would be saved right now?
        status, can_save = "", False
        if mode is None:
            status = "Press 1, 2, 3 or B to pick what you are holding"
        elif mode == "bg":
            can_save = not dets                    # never save a sign as "background"
            status = "OK - no sign in view" if can_save else "A sign is visible - put it away for background"
        elif best and best[1] >= MIN_CONF:
            can_save = True
            status = f"OK - sign found ({best[1]:.2f})"
        else:
            status = "No sign found yet - move it closer / steadier"

        now = time.time()
        if saving and can_save and now - last_save >= SAVE_EVERY:
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame)       # CLEAN picture
            if mode == "bg":
                (out / "labels" / f"{name}.txt").write_text("")
            else:
                x1, y1, x2, y2 = best[0]
                xc, yc, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
                (out / "labels" / f"{name}.txt").write_text(f"{mode} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            counts[mode] += 1
            last_save = now

        # ---- preview (drawn on a COPY, never on the saved picture) ----
        view = frame.copy()
        if best and mode not in (None, "bg"):
            x1, y1, x2, y2 = map(int, best[0])
            cv2.rectangle(view, (x1, y1), (x2, y2), COLORS[mode], 3)
            cv2.putText(view, f"will save as {MODE_TEXT[mode]}", (x1, max(70, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[mode], 2)
        cv2.rectangle(view, (0, 0), (w, 70), (0, 0, 0), -1)
        rec = "SAVING" if saving else "paused (SPACE to save)"
        cv2.putText(view, f"Holding: {MODE_TEXT[mode]}   |   {rec}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255) if saving else (255, 255, 255), 2)
        cv2.putText(view, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0) if can_save else (0, 200, 255), 2)
        tally = (f"35: {counts[0]}/{GOAL_PER_SIGN}   45: {counts[1]}/{GOAL_PER_SIGN}   "
                 f"STOP: {counts[2]}/{GOAL_PER_SIGN}   background: {counts['bg']}/{GOAL_BACKGROUND}")
        cv2.rectangle(view, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(view, tally, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in MODES:
            mode, saving = MODES[key], False         # pause when switching signs
            print(f"Now holding: {MODE_TEXT[mode]}  (press SPACE to start saving)")
        elif key == ord(" "):
            saving = not saving and mode is not None
            print("Saving ON" if saving else "Saving paused")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    total = sum(counts.values())
    print(f"\nSaved so far: 35={counts[0]}  45={counts[1]}  STOP={counts[2]}  background={counts['bg']}  (total {total})")
    print(f"Folder: {out}")
    print("Next: right-click the my_signs folder > Compress to ZIP, upload my_signs.zip to Google Drive,")
    print("then run Train_YOLO26n_Signs_Colab.ipynb again.")


if __name__ == "__main__":
    main()
