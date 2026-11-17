"""
Real-time traffic sign detection (stop sign, 35 mph, 45 mph) with a YOLO model,
with optional FGSM / PGD / noise attacks applied to each frame for robustness testing.

Setup (one time, in Command Prompt):
    py -m pip install ultralytics opencv-python

Easiest way to run:
    Put this file, attacks.py and your model (.pt) file in the same folder,
    open this file in IDLE, and press F5. It finds the model automatically and
    opens your webcam.

To change settings, edit the SETTINGS section below. For example, set
SOURCE = "drive.mp4" to run on a video file instead of the webcam, or
ATTACK = "fgsm" to start with the attack already on.

Keys (click the video window first):
    A       cycle attack: off -> noise -> fgsm -> pgd -> cw (Carlini-Wagner)
    G       switch goal: hide (signs vanish) / swap (stop -> 45, 45 -> 35, 35 -> 45)
    [ ]     make the attack weaker / stronger (epsilon -1 / +1 out of 255; not used by C&W)
    S       show side-by-side (clean | attacked) or attacked only
    R       reset the "fooled" counter
    Q       quit (or close the window)

The attacks only change the frames inside this program, using attacks.py, to
test how robust your own model is. Nothing outside the program is changed.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from attacks import ATTACKS, GOALS, SignModel, sign_kind

# ------------------------------- SETTINGS --------------------------------
WEIGHTS = ""    # leave empty to auto-find a .pt file, or put a full path here
SOURCE = "0"    # "0" = built-in webcam, "1" = USB camera, or a video like "drive.mp4"
CONF = 0.6      # how sure the model must be to draw a box; raise to 0.7 if it boxes your face
IMGSZ = 416     # 416 = good balance; 320 is faster but blurrier (more wrong boxes); 640 = most accurate
ATTACK = "off"  # starting attack: "off", "noise", "fgsm", "pgd" or "cw" (press A to change)
GOAL = "hide"   # "hide" = make signs disappear, "swap" = make a sign read as another sign (press G)
EPS_255 = 8     # attack strength in brightness levels out of 255 (8/255 ~= 0.03)
PGD_STEPS = 5   # more steps = stronger PGD but slower (attack_eval.py uses 10)
CW_LIVE_STEPS = 10  # C&W steps per frame (more = smaller change but slower; attack_eval.py uses 20)
SIDE_BY_SIDE = True
# --- speed settings ---
PROCESS_WIDTH = 640    # frames wider than this are shrunk (faster, window fits the screen); 0 = never shrink
ATTACKED_CONF = 0.3     # attacked side: also show weaker guesses (30%+) sitting on the sign,
                        # so a half-finished swap shows as "35 (weak)" instead of nothing
SKIP_IF_NO_SIGN = True  # don't run the attack on frames with no sign (nothing to fool there)
DEVICE = "auto"         # "auto" uses an NVIDIA GPU if PyTorch can see one, else "cpu"
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
        # Try the camera drivers in turn and keep the first one that gives a real
        # (not black) picture. The camera's own normal resolution is used.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None] if sys.platform == "win32" else [None]
        for backend in backends:
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue
            for _ in range(15):                 # cameras often send a few dark frames while starting
                ok, frame = cap.read()
                if ok and frame is not None and frame.mean() > 8:
                    return cap
            cap.release()
        sys.exit(f"Camera {index} isn't giving a picture (black or no frames).\n"
                 "Close other apps or other copies of this program using the webcam, check the\n"
                 "camera cover / camera key, and check Settings > Privacy & security > Camera.\n"
                 "Or try SOURCE = \"1\".")

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


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def outcome(ref, adv_dets, names):
    """Same rule as attack_eval.py. For a sign found on the clean frame:
    'kept'    - attacked frame still has a box on it with the SAME sign type
    'flipped' - attacked frame has a box on it with a DIFFERENT sign type (e.g. 45 -> 35)
    'missed'  - no box on it at all (the sign disappeared)"""
    kind = sign_kind(names[ref[5]])
    best, best_kind = 0.0, None
    for a in adv_dets:
        v = iou(ref, a)
        if v > best:
            best, best_kind = v, sign_kind(names[a[5]])
    if best < 0.5:
        return "missed"
    return "kept" if best_kind == kind else "flipped"


def new_counts():
    return {"kept": 0, "flipped": 0, "missed": 0}


def success_rate(c, goal):
    """Attack success rate:
    hide - sign no longer read correctly (missed or flipped) / signs found on the clean frame
    swap - sign read as a DIFFERENT sign (flipped)          / signs found on the clean frame"""
    n = sum(c.values())
    if not n:
        return None
    return (c["flipped"] if goal == "swap" else c["flipped"] + c["missed"]) / n


def counts_text(c, goal=None):
    n = sum(c.values())
    if not n:
        return "no signs tested yet"
    pct = lambda k: f"{c[k] / n:.0%}"
    text = f"flipped {c['flipped']} ({pct('flipped')})  missed {c['missed']} ({pct('missed')})  kept {c['kept']} ({pct('kept')})  of {n}"
    if goal:
        text = f"SUCCESS {success_rate(c, goal):.0%}  |  " + text
    return text


def draw_frame(frame, dets, names, title, title_color, fps=None, sure=0.0):
    """Draw boxes, the alert banner and a title. Boxes below `sure` are drawn thin and
    labelled "(weak)". Returns the set of class names seen."""
    seen = set()
    for x1, y1, x2, y2, s, c in dets:
        name = names[c]
        weak = s < sure
        if not weak:
            seen.add(name)
        col = COLORS.get(name, (0, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1 if weak else 3)
        cv2.putText(frame, f"{name} {s:.2f}" + (" (weak)" if weak else ""), (x1, max(60, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    alert = next(((text, COLORS[cls]) for cls, text in ALERTS if cls in seen), None)

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    x = 10
    if fps is not None:
        cv2.putText(frame, f"{fps:.1f} FPS", (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        x = 160
    cv2.putText(frame, title, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, title_color, 2)
    if alert:
        cv2.putText(frame, alert[0], (frame.shape[1] - 300, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert[1], 2)
    return seen


def main():
    ap = argparse.ArgumentParser(description="Real-time traffic sign detection with attacks")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--attack", default=ATTACK, choices=ATTACKS)
    ap.add_argument("--goal", default=GOAL, choices=GOALS)
    ap.add_argument("--eps", type=int, default=EPS_255, help="attack strength out of 255")
    args, _ = ap.parse_known_args()

    weights = find_weights(args.weights)
    print(f"Loading model: {weights}")
    model = SignModel(weights, imgsz=args.imgsz)   # same model, wrapped so attacks can use it
    names = model.names
    print(f"Classes: {list(names.values())}")

    device = DEVICE
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.net.to(device)
    print(f"Running on: {device.upper()}"
          + ("" if device == "cuda" else "  (no NVIDIA GPU found by PyTorch - CPU is slower)"))

    attack, goal, eps255 = args.attack, args.goal, args.eps
    side_by_side = SIDE_BY_SIDE
    counts = new_counts()

    cap = open_source(str(args.source))
    window = "Traffic Sign Detection (Q quit, A attack, G goal, [ ] strength)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Running... press Q in the video window (or close it) to stop.")
    print("Press A to turn on an attack.")

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
        if PROCESS_WIDTH and frame.shape[1] > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(round(frame.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)

        eps = eps255 / 255.0

        # Clean detection (same input tensor the attack starts from).
        x, meta = model.to_tensor(frame)
        x = x.to(device)
        meta["mask"] = meta["mask"].to(device)
        maybe = model.detect(x, meta, conf=0.25)            # anything that might be a sign
        clean = [d for d in maybe if d[4] >= args.conf]     # signs the model is sure about (boxes drawn)

        # Attacked detection.
        attack_on = attack != "off" and (eps > 0 or attack == "cw")
        change_text = ""
        if attack_on and SKIP_IF_NO_SIGN and not maybe:
            # No sign in view: nothing to fool, so skip the slow attack on this frame.
            attacked_frame, attacked = frame.copy(), []
        elif attack_on:
            try:
                x_adv = model.attack(attack, x, meta, eps, goal, PGD_STEPS, CW_LIVE_STEPS)
            except ValueError as e:     # e.g. swap needs both 35 and 45 classes
                print(e)
                goal = "hide"
                x_adv = model.attack(attack, x, meta, eps, goal, PGD_STEPS, CW_LIVE_STEPS)
            if attack == "cw":          # C&W chooses its own size: show how big it ended up
                d = (x_adv - x).abs()
                change_text = f"  change: max {d.max().item() * 255:.0f}/255, avg {d.mean().item() * 255:.2f}/255"
            attacked_frame = model.add_perturbation(frame, x.cpu(), x_adv.cpu(), meta)
            adv_all = model.detect(x_adv, meta, conf=min(ATTACKED_CONF, args.conf))
            # keep sure boxes, plus weaker boxes only where a real sign is (not on your face)
            attacked = [a for a in adv_all
                        if a[4] >= args.conf or any(iou(a, d) >= 0.5 for d in clean)]
            for d in clean:
                counts[outcome(d, attacked, names)] += 1

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6))
        t_prev = now
        frames += 1

        if not attack_on:
            draw_frame(frame, clean, names, "attack OFF (press A)", (200, 200, 200), fps)
            shown = frame
        else:
            strength = "smallest change" if attack == "cw" else f"eps={eps255}/255"
            label = f"{attack.upper()} {goal} {strength}" + ("" if maybe or not SKIP_IF_NO_SIGN else " (no sign - waiting)")
            draw_frame(attacked_frame, attacked, names, label, (0, 200, 255), None if side_by_side else fps,
                       sure=args.conf)
            h_img = attacked_frame.shape[0]
            cv2.putText(attacked_frame, counts_text(counts), (10, h_img - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            rate = success_rate(counts, goal)
            if rate is not None:
                cv2.putText(attacked_frame, f"ATTACK SUCCESS: {rate:.0%}", (10, h_img - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if rate >= 0.5 else (0, 165, 255), 2)
            if change_text:
                cv2.putText(attacked_frame, change_text.strip(), (10, h_img - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            if side_by_side:
                draw_frame(frame, clean, names, "CLEAN", (255, 255, 255), fps)
                shown = np.hstack([frame, attacked_frame])
            else:
                shown = attacked_frame

        cv2.imshow(window, shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        changed = True
        old = (attack, goal, eps255)
        if key in (ord("a"), ord("A")):
            attack = ATTACKS[(ATTACKS.index(attack) + 1) % len(ATTACKS)]
        elif key in (ord("g"), ord("G")):
            goal = GOALS[(GOALS.index(goal) + 1) % len(GOALS)]
        elif key == ord("]"):
            eps255 = min(77, eps255 + 1)    # 77/255 ~= 0.30, the top of attack_eval.py's sweep
        elif key == ord("["):
            eps255 = max(0, eps255 - 1)
        elif key in (ord("s"), ord("S")):
            side_by_side = not side_by_side
            changed = False
        elif key in (ord("r"), ord("R")):
            pass
        else:
            changed = False
        if changed:                         # new setting = new measurement
            if sum(counts.values()) and old[0] != "off":   # log the result of the setting we're leaving
                print(f"   RESULT {old[0]} {old[1]} {'(C&W picks size)' if old[0] == 'cw' else f'eps {old[2]}/255'}: {counts_text(counts, old[1])}")
            counts = new_counts()
            print(f"Attack: {attack}  goal: {goal}  eps: {eps255}/255 ({eps255 / 255:.3f})")
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t_start
    if frames and elapsed > 0:
        print(f"Processed {frames} frames, average {frames / elapsed:.1f} FPS.")
    if sum(counts.values()):
        print(f"   RESULT {attack} {goal} {'(C&W picks size)' if attack == 'cw' else f'eps {eps255}/255'}: {counts_text(counts, goal)}")


if __name__ == "__main__":
    main()
