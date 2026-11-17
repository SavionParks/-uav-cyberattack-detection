"""
attack_eval.py - measure how FGSM and PGD attacks affect the sign detector,
using the same epsilon sweep as the team's vehicle classifier (0.00-0.30, 8 steps).

Put this file, attacks.py and your model (.pt) in one folder. Put test pictures
(photos of stop / 35 / 45 mph signs) in a folder called "test_images" next to it,
or point SOURCE at a video. Then open in IDLE and press F5.

How it scores: the detections on the clean image are the reference. After each
attack, every reference sign is counted as
    kept    - still detected, same sign type
    flipped - detected as a DIFFERENT sign (e.g. 45 read as 35)
    missed  - not detected at all
and any new detection that wasn't there before is a "phantom".

Output (in the "attack_results" folder):
    results.csv          every number, per attack / epsilon / sign type
    detection_rate.png   chart like the team's Figure 2
    attack_success.png   attack success rate vs epsilon (hide: sign lost; swap: sign read as another sign)
    flip_rate.png        (swap goal) how often a sign is read as a different sign
    examples_eps*.png    clean vs noise vs FGSM vs PGD vs C&W on one test image
    results_cw_*.csv     C&W (Carlini-Wagner L2) results, with how big its change was

C&W has no epsilon: it searches for the smallest change that fools the model, so it is
run once (not swept) and shown as a flat dashed line on the charts.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from attacks import CW_STEPS, SignModel, sign_kind

# ------------------------------- SETTINGS --------------------------------
WEIGHTS = ""                  # empty = auto-find the .pt file next to this script / in Downloads
SOURCE = "test_images"        # a folder of pictures, or a video file (e.g. "drive.mp4")
GOAL = "hide"                 # "hide" = make signs disappear, "swap" = make a sign read as another sign
EPSILONS = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30]   # same sweep as the team's report
PGD_STEPS = 10
CONF = 0.35                   # confidence needed to count as a detection
IMGSZ = 640
VIDEO_EVERY = 15              # for videos: test every Nth frame
MAX_IMAGES = 40               # cap so the run finishes in reasonable time
EXAMPLE_EPS = 0.03            # epsilon used for the example picture
OUT_DIR = "attack_results"
RUN_CW = True                 # also run Carlini-Wagner L2 (slower: CW_STEPS steps per image)
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ATTACK_KINDS = ["noise", "fgsm", "pgd"]
KIND_COLOR = {"STOP": (0, 0, 255), "35": (0, 200, 0), "45": (0, 140, 255)}


def find_weights(choice):
    if choice:
        for p in (Path(choice), SCRIPT_DIR / choice):
            if p.is_file():
                return p
        sys.exit(f"Could not find the model file: {choice}")
    found = []
    for d in (SCRIPT_DIR, Path.home() / "Downloads"):
        if d.is_dir():
            found += [f for f in d.glob("*.pt") if not f.name.lower().startswith("yolo")]
    if not found:
        sys.exit("No model (.pt) file found. Put it next to this script or set WEIGHTS.")
    return max(found, key=lambda f: ("best" in f.name.lower() or "stop_sign" in f.name.lower(), f.stat().st_mtime))


def load_images(source):
    p = Path(source)
    if not p.exists():
        p = SCRIPT_DIR / source
    if not p.exists():
        sys.exit(f"Could not find '{source}'. Make a folder called test_images next to this script "
                 f"and put sign photos in it, or set SOURCE to a video file.")
    images = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                img = cv2.imread(str(f))
                if img is not None:
                    images.append((f.name, img))
    else:
        cap = cv2.VideoCapture(str(p))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % VIDEO_EVERY == 0:
                images.append((f"frame_{i:05d}", frame))
            i += 1
        cap.release()
    if not images:
        sys.exit(f"No images found in {p}")
    return images[:MAX_IMAGES]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(ref, adv):
    """Match clean detections to attacked ones. Returns per-sign outcomes and phantom count."""
    outcomes, used = [], set()
    for r in ref:
        best, best_j = 0.0, None
        for j, a in enumerate(adv):
            if j in used:
                continue
            v = iou(r[:4], a[:4])
            if v > best:
                best, best_j = v, j
        if best_j is not None and best >= 0.5:
            used.add(best_j)
            outcomes.append((r[4], "kept" if adv[best_j][4] == r[4] else "flipped"))
        else:
            outcomes.append((r[4], "missed"))
    return outcomes, len(adv) - len(used)


def success(st, goal):
    """Attack success rate. hide: sign no longer read correctly (missed or flipped).
    swap: sign read as a different sign (flipped). Both divided by signs found on the clean image."""
    n = st["signs"]
    return (st["flipped"] if goal == "swap" else st["flipped"] + st["missed"]) / n if n else 0.0


def draw(img, dets, title):
    img = img.copy()
    for x1, y1, x2, y2, kind, s in dets:
        col = KIND_COLOR.get(kind, (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, img.shape[1] // 300))
        cv2.putText(img, f"{kind} {s:.2f}", (x1, max(25, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.6, img.shape[1] / 1400), col, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="FGSM / PGD evaluation for the sign detector")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--goal", default=GOAL, choices=["hide", "swap"])
    args, _ = ap.parse_known_args()

    model = SignModel(find_weights(args.weights), IMGSZ)
    out_dir = SCRIPT_DIR / OUT_DIR
    out_dir.mkdir(exist_ok=True)
    images = load_images(args.source)

    def run(x, meta):
        return [(x1, y1, x2, y2, sign_kind(model.names[c]), s)
                for x1, y1, x2, y2, s, c in model.detect(x, meta, conf=CONF)]

    # clean pass: keep only images where the model finds at least one sign
    prepared = []
    for name, img in images:
        x, meta = model.to_tensor(img)
        ref = run(x, meta)
        if ref:
            prepared.append((name, img, x, meta, ref))
    if not prepared:
        sys.exit("The model didn't detect any signs in the clean images, so there is nothing to attack.\n"
                 "Use clearer or closer sign photos.")
    n_signs = sum(len(p[4]) for p in prepared)
    print(f"{len(prepared)} of {len(images)} images have detections ({n_signs} signs). Goal: {args.goal.upper()}")

    # stats[(attack, eps, kind)] = counts
    stats = {}
    example = prepared[0]
    example_imgs = {}
    t0 = time.time()
    total = len(ATTACK_KINDS) * len(EPSILONS)
    done = 0
    for kind in ATTACK_KINDS:
        for eps in EPSILONS:
            for idx, (name, img, x, meta, ref) in enumerate(prepared):
                x_adv = model.attack(kind, x, meta, eps, args.goal, PGD_STEPS) if eps > 0 else x
                adv = run(x_adv, meta)
                outcomes, phantoms = compare(ref, adv)
                for sk in {"ALL"} | {o[0] for o in outcomes}:
                    st = stats.setdefault((kind, eps, sk), dict(signs=0, kept=0, flipped=0, missed=0, phantoms=0, images=0))
                    for k2, res in outcomes:
                        if sk in ("ALL", k2):
                            st["signs"] += 1
                            st[res] += 1
                    if sk == "ALL":
                        st["phantoms"] += phantoms
                        st["images"] += 1
                if idx == 0 and abs(eps - EXAMPLE_EPS) < 1e-9:
                    shown = model.add_perturbation(img, x, x_adv, meta)
                    example_imgs[kind] = draw(shown, adv, f"{kind.upper()} eps={eps:.2f}")
            done += 1
            st = stats[(kind, eps, "ALL")]
            print(f"[{done:2d}/{total}] {kind:5s} eps={eps:.2f}  SUCCESS {success(st, args.goal):6.1%}"
                  f"  detected {st['kept'] / st['signs']:6.1%}"
                  f"  flipped {st['flipped'] / st['signs']:6.1%}  missed {st['missed'] / st['signs']:6.1%}"
                  f"  ({time.time() - t0:.0f}s)")

    # ---- CSV ----
    csv_path = out_dir / f"results_{args.goal}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["goal", "attack", "epsilon", "sign_type", "signs", "attack_success_rate", "detection_rate",
                    "flip_rate", "miss_rate", "phantoms_per_image"])
        for (kind, eps, sk), st in sorted(stats.items(), key=lambda t: (t[0][0], t[0][1], t[0][2])):
            n = st["signs"]
            w.writerow([args.goal, kind, eps, sk, n, round(success(st, args.goal), 4),
                        round(st["kept"] / n, 4), round(st["flipped"] / n, 4),
                        round(st["missed"] / n, 4),
                        round(st["phantoms"] / st["images"], 3) if sk == "ALL" else ""])
    print(f"Saved {csv_path}")

    # ---- C&W: run once (it chooses its own size) ----
    cw = None
    if RUN_CW:
        cw = dict(signs=0, kept=0, flipped=0, missed=0, phantoms=0, images=0, l2=[], max255=[], avg255=[])
        cw_by_kind = {}
        for idx, (name, img, x, meta, ref) in enumerate(prepared):
            x_adv = model.cw(x, meta, args.goal)
            adv = run(x_adv, meta)
            outcomes, phantoms = compare(ref, adv)
            for k2, res in outcomes:
                cw["signs"] += 1; cw[res] += 1
                kk = cw_by_kind.setdefault(k2, dict(signs=0, kept=0, flipped=0, missed=0))
                kk["signs"] += 1; kk[res] += 1
            cw["phantoms"] += phantoms; cw["images"] += 1
            d = (x_adv - x) * meta["mask"]
            cw["l2"].append(float(d.pow(2).sum().sqrt()))
            cw["max255"].append(float(d.abs().max()) * 255)
            cw["avg255"].append(float(d.abs().sum() / (meta["mask"].sum())) * 255)
            if idx == 0:
                shown = model.add_perturbation(img, x, x_adv, meta)
                example_imgs["cw"] = draw(shown, adv, "C&W (smallest change)")
        n = cw["signs"]
        print(f"[C&W] {CW_STEPS} steps  SUCCESS {success(cw, args.goal):6.1%}  detected {cw['kept'] / n:6.1%}  flipped {cw['flipped'] / n:6.1%}"
              f"  missed {cw['missed'] / n:6.1%}  avg change {np.mean(cw['avg255']):.2f}/255"
              f"  (max {np.mean(cw['max255']):.1f}/255)  ({time.time() - t0:.0f}s)")
        cw_path = out_dir / f"results_cw_{args.goal}.csv"
        with open(cw_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["goal", "attack", "sign_type", "signs", "attack_success_rate", "detection_rate",
                        "flip_rate", "miss_rate",
                        "avg_pixel_change_255", "avg_max_pixel_change_255", "avg_l2"])
            rows = [("ALL", cw)] + sorted(cw_by_kind.items())
            for sk, st in rows:
                m = st["signs"]
                w.writerow([args.goal, "cw", sk, m, round(success(st, args.goal), 4), round(st["kept"] / m, 4), round(st["flipped"] / m, 4),
                            round(st["missed"] / m, 4),
                            round(float(np.mean(cw["avg255"])), 3) if sk == "ALL" else "",
                            round(float(np.mean(cw["max255"])), 2) if sk == "ALL" else "",
                            round(float(np.mean(cw["l2"])), 4) if sk == "ALL" else ""])
        print(f"Saved {cw_path}")

    # ---- charts ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    styles = {"noise": ("gray", ":", "o"), "fgsm": ("tab:blue", "-", "s"), "pgd": ("tab:red", "--", "^")}
    charts = [("attack_success", "success", "Attack success rate"),
              ("detection_rate", "kept", "Signs still detected correctly")]
    if args.goal == "swap":
        charts.append(("flip_rate", "flipped", "Signs read as a different sign"))
    for fname, key, ylabel in charts:
        plt.figure(figsize=(8, 5))
        for kind in ATTACK_KINDS:
            ys = [success(stats[(kind, e, "ALL")], args.goal) if key == "success" else
                  stats[(kind, e, "ALL")][key] / stats[(kind, e, "ALL")]["signs"] for e in EPSILONS]
            c, ls, mk = styles[kind]
            plt.plot(EPSILONS, ys, color=c, linestyle=ls, marker=mk, label=kind.upper())
        if cw:
            plt.axhline(success(cw, args.goal) if key == "success" else cw[key] / cw["signs"],
                        color="tab:purple", linestyle="-.",
                        label=f"C&W (no eps, avg change {np.mean(cw['avg255']):.1f}/255)")
        plt.title(f"YOLO sign detector under attack ({args.goal.upper()} goal, {n_signs} signs)")
        plt.xlabel("Epsilon (attack strength)")
        plt.ylabel(ylabel)
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"{fname}_{args.goal}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved {path}")

    # ---- example picture ----
    if example_imgs:
        name, img, x, meta, ref = example
        panels = [draw(img, ref, "CLEAN")] + [example_imgs[k] for k in ATTACK_KINDS + ["cw"] if k in example_imgs]
        h = 360
        panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
        if len(panels) % 2:                       # pad to an even count so it fits 2 rows
            panels.append(np.zeros_like(panels[0]))
        half = len(panels) // 2
        grid = np.vstack([np.hstack(panels[:half]), np.hstack(panels[half:])])
        path = out_dir / f"examples_{args.goal}_eps{EXAMPLE_EPS:.2f}.png"
        cv2.imwrite(str(path), grid)
        print(f"Saved {path}")

    print("Done.")


if __name__ == "__main__":
    main()
