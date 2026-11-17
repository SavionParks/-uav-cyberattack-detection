"""
attacks.py - shared code for the traffic sign detector.

Contains:
  * SignModel: loads the YOLO .pt file and runs detection on a frame, using the
    exact same input tensor that the attacks modify (so results are consistent).
  * sign_kind(): turns any class name into "STOP", "35" or "45", so models whose
    classes are named "35_speed_limit" or "speed_limit_35" both work.
  * Three perturbation methods, all limited to a maximum change of `eps` per pixel
    (pixel values are 0-1, so eps=0.03 means about 8 out of 255 brightness levels):
      - noise: random +/- eps noise (a control: "is it just noise?")
      - fgsm:  Fast Gradient Sign Method, one gradient step
      - pgd:   Projected Gradient Descent, several smaller gradient steps
    plus
      - cw:    Carlini-Wagner L2, searches for the SMALLEST change that fools the model
               (no eps limit; see the CW_ settings)
    and two goals:
      - "hide": make the detector miss signs
      - "swap": make a sign read as a different sign (stop -> 45, 45 -> 35, 35 -> 45; see SWAP_TO)

These are digital attacks on your own model, for robustness testing only.
"""
import math

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # older ultralytics versions
    from ultralytics.utils.ops import non_max_suppression

ATTACKS = ["off", "noise", "fgsm", "pgd", "cw"]
GOALS = ["hide", "swap"]

# ---- Carlini-Wagner (C&W) L2 settings ----
CW_STEPS = 20        # optimisation steps per image (more = smaller change, slower)
CW_C = 2.0           # how much "fool the model" matters vs. "keep the change small"
CW_LR = 0.02         # step size of the optimiser
CW_KAPPA = 0.5       # swap: how far the new sign must beat the real one
CW_HIDE_SCORE = 0.2  # hide: push every sign score below this
CW_SWAP_SCORE = 0.6  # swap: push the new sign above this (matches the camera's CONF)

# What each sign should turn into for the "swap" goal. Change these if you like,
# e.g. "STOP": "35". Any sign not listed here is turned into the model's 2nd-best guess.
SWAP_TO = {
    "STOP": "45",   # stop sign  -> 45 mph
    "45": "35",     # 45 mph     -> 35 mph
    "35": "45",     # 35 mph     -> 45 mph
}
SWAP_TARGET_WEIGHT = 3.0   # how hard swap pushes the NEW sign up vs. the real sign down


def _logit(p):
    return math.log(p / (1 - p))


def sign_kind(name):
    """Map a class name to 'STOP', '35', '45', or the name itself."""
    n = name.lower()
    if "stop" in n:
        return "STOP"
    if "35" in n:
        return "35"
    if "45" in n:
        return "45"
    return name


class SignModel:
    def __init__(self, weights, imgsz=640):
        self.yolo = YOLO(str(weights))
        self.net = self.yolo.model.float().eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        # YOLO26 (and YOLOv10) are "end-to-end": by default they output a short finished
        # list of boxes instead of the raw per-spot class scores the attacks need.
        # Switch them to their regular head, which gives the same raw scores as YOLO11.
        head = self.net.model[-1] if hasattr(self.net, "model") else None
        if head is not None and getattr(head, "end2end", False):
            try:
                head.end2end = False
                print("YOLO26-style model: using its regular (raw score) head for the attacks.")
            except Exception:
                pass
        self.names = self.yolo.names
        self.imgsz = int(imgsz)
        self.kind_to_idx = {sign_kind(v): k for k, v in self.names.items()}
        self._check_output()

    def _check_output(self):
        """Make sure the model gives raw scores shaped (1, 4 + classes, spots)."""
        with torch.no_grad():
            pred, _ = self.class_scores(torch.zeros(1, 3, 320, 320))
        nc = len(self.names)
        if pred.dim() != 3 or pred.shape[1] != 4 + nc:
            raise RuntimeError(
                f"This model's output has shape {tuple(pred.shape)}, but the attacks expect "
                f"(1, {4 + nc}, N). Send this message to get attacks.py updated for this model.")

    # ----- image <-> tensor -------------------------------------------------
    def to_tensor(self, frame):
        """Letterbox a BGR frame into a (1,3,S,S) RGB tensor in 0-1, plus layout info."""
        h, w = frame.shape[:2]
        s = self.imgsz
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        left, top = (s - nw) // 2, (s - nh) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255.0
        mask = torch.zeros_like(x)
        mask[..., top:top + nh, left:left + nw] = 1.0  # only perturb the real image, not padding
        meta = dict(r=r, left=left, top=top, nw=nw, nh=nh, w=w, h=h, mask=mask)
        return x, meta

    def add_perturbation(self, frame, x_clean, x_adv, meta):
        """Draw the perturbation onto the full-size frame so the display stays sharp."""
        m = meta
        delta = (x_adv - x_clean)[0, :, m["top"]:m["top"] + m["nh"], m["left"]:m["left"] + m["nw"]]
        delta = delta.permute(1, 2, 0).numpy()[:, :, ::-1] * 255.0
        delta = cv2.resize(delta, (m["w"], m["h"]), interpolation=cv2.INTER_NEAREST)
        return np.clip(frame.astype(np.float32) + delta, 0, 255).astype(np.uint8)

    # ----- model ------------------------------------------------------------
    def class_scores(self, x):
        out = self.net(x)
        pred = out[0] if isinstance(out, (list, tuple)) else out
        return pred, pred[:, 4:, :]  # (1, 4+nc, anchors), (1, nc, anchors)

    def detect(self, x, meta, conf=0.25, iou=0.45):
        """Return a list of detections: (x1, y1, x2, y2, score, class_index)."""
        with torch.no_grad():
            pred, _ = self.class_scores(x)
            # agnostic=True: one label per sign, so a sign can't be both 35 AND 45
            det = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, agnostic=True, max_det=50)[0]
        out = []
        m = meta
        for x1, y1, x2, y2, s, c in det.tolist():
            x1 = (x1 - m["left"]) / m["r"]; x2 = (x2 - m["left"]) / m["r"]
            y1 = (y1 - m["top"]) / m["r"]; y2 = (y2 - m["top"]) / m["r"]
            out.append((max(0, int(x1)), max(0, int(y1)), min(m["w"] - 1, int(x2)), min(m["h"] - 1, int(y2)),
                        float(s), int(c)))
        return out

    # ----- attacks ----------------------------------------------------------
    def _swap_pair(self, x):
        """Decide the swap from the clean image: find the sign in view and pick what to turn it into
        (see SWAP_TO). Returns None if no sign is in view."""
        if len(self.names) < 2:
            raise ValueError("Swap attack needs a model with at least two classes.")
        with torch.no_grad():
            _, cls = self.class_scores(x)
        best_per_class = cls[0].max(dim=1).values          # strongest score of each class anywhere
        src = int(best_per_class.argmax())
        if best_per_class[src] < 0.25:
            return None  # no sign in view: nothing to swap
        tgt = self.kind_to_idx.get(SWAP_TO.get(sign_kind(self.names[src])))
        if tgt is None or tgt == src:
            # sign type not in SWAP_TO (or target class missing): use the model's 2nd-best guess
            order = best_per_class.argsort(descending=True).tolist()
            tgt = next(i for i in order if i != src)
        # Only the spots that are actually ON the sign (score at least half of the best spot).
        # Before, it took the 50 best spots even if some were far away on the wall or your
        # hand, so the swap made new "35"/"45" boxes pop up in random places.
        top = cls[0, src].topk(min(50, cls.shape[2]))
        anchors = top.indices[top.values >= 0.5 * top.values[0]]
        return src, tgt, anchors

    def _loss(self, x, goal, pair=None):
        """A number the attack tries to make smaller."""
        _, cls = self.class_scores(x)
        cls = cls[0]  # (nc, anchors)
        if goal == "hide":
            # push down the strongest sign scores anywhere in the image
            return cls.flatten().topk(min(100, cls.numel())).values.sum()
        # "swap": on the sign itself, train the scores toward "target = yes, real sign = no".
        # The model's scores are already squashed to 0-1 (sigmoid). Subtracting those
        # directly gave almost no push to the target class while it was near 0, so the
        # attack mostly just lowered the real sign and the box vanished instead of flipping.
        # Undoing the squash (logit) and using a cross-entropy loss gives the target class
        # a strong push even when its score starts near zero.
        src, tgt, a = pair
        z_src = torch.logit(cls[src, a], eps=1e-6)
        z_tgt = torch.logit(cls[tgt, a], eps=1e-6)
        # The "raise the target" part is weighted 3x so a single FGSM step mostly builds up the
        # new sign instead of just erasing the old one (which made the box vanish).
        return (torch.nn.functional.softplus(z_src) + SWAP_TARGET_WEIGHT * torch.nn.functional.softplus(-z_tgt)).sum()

    def _grad(self, x, goal, pair=None):
        x = x.clone().requires_grad_(True)
        loss = self._loss(x, goal, pair)
        loss.backward()
        return x.grad.detach()

    @staticmethod
    def _finish(x_adv):
        # round to real 8-bit pixel values, as if the image were saved and reloaded
        return torch.round(x_adv.clamp(0, 1) * 255) / 255

    def noise(self, x, meta, eps):
        rnd = torch.sign(torch.rand_like(x) - 0.5)
        return self._finish(x + eps * rnd * meta["mask"])

    def fgsm(self, x, meta, eps, goal="hide"):
        pair = self._swap_pair(x) if goal == "swap" else None
        if goal == "swap" and pair is None:
            return x
        g = self._grad(x, goal, pair)
        return self._finish(x - eps * g.sign() * meta["mask"])

    def pgd(self, x, meta, eps, goal="hide", steps=10):
        pair = self._swap_pair(x) if goal == "swap" else None
        if goal == "swap" and pair is None:
            return x
        alpha = 2.5 * eps / steps
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps) * meta["mask"]).clamp(0, 1)
        for _ in range(steps):
            g = self._grad(x_adv, goal, pair)
            x_adv = x_adv - alpha * g.sign() * meta["mask"]
            x_adv = (x + (x_adv - x).clamp(-eps, eps)).clamp(0, 1)
        return self._finish(x_adv)

    def _cw_fooled_score(self, x, goal, pair=None):
        """C&W's 'how far from fooled' number: 0 once the goal is reached, bigger = further away.
        hide: every sign score on the image must drop below CW_HIDE_SCORE.
        swap: on the sign, the new sign must beat the real one AND reach CW_SWAP_SCORE."""
        _, cls = self.class_scores(x)
        cls = cls[0]
        if goal == "hide":
            top = torch.logit(cls.flatten().topk(min(100, cls.numel())).values, eps=1e-6)
            return torch.relu(top - _logit(CW_HIDE_SCORE)).sum()
        src, tgt, a = pair
        z_src = torch.logit(cls[src, a], eps=1e-6)
        z_tgt = torch.logit(cls[tgt, a], eps=1e-6)
        beat = torch.relu(z_src - z_tgt + CW_KAPPA)
        sure = torch.relu(_logit(CW_SWAP_SCORE) - z_tgt)
        return (beat + sure).mean() * len(a) ** 0.5

    def cw(self, x, meta, goal="hide", steps=None, c=None, lr=None):
        """Carlini-Wagner L2: search for the SMALLEST change that fools the model.
        There is no epsilon limit; instead it minimises   size_of_change^2 + c * how_far_from_fooled.
        Works in tanh space so pixels always stay inside 0-1."""
        steps = steps or CW_STEPS
        c = CW_C if c is None else c
        lr = lr or CW_LR
        pair = self._swap_pair(x) if goal == "swap" else None
        if goal == "swap" and pair is None:
            return x
        mask = meta["mask"]
        w = torch.atanh((x * 2 - 1).clamp(-1 + 1e-6, 1 - 1e-6)).detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr)
        best, best_size = None, float("inf")
        for _ in range(steps):
            cand = 0.5 * (torch.tanh(w) + 1)
            x_adv = x * (1 - mask) + cand * mask          # never change the grey padding
            size = ((x_adv - x) ** 2).sum()
            f = self._cw_fooled_score(x_adv, goal, pair)
            loss = size + c * f
            opt.zero_grad()
            loss.backward()
            opt.step()
            if f.item() == 0 and size.item() < best_size:  # fooled: remember the smallest such change
                best, best_size = x_adv.detach().clone(), size.item()
        if best is None:
            with torch.no_grad():
                best = x * (1 - mask) + 0.5 * (torch.tanh(w) + 1) * mask
        return self._finish(best)

    def attack(self, kind, x, meta, eps, goal="hide", steps=10, cw_steps=None):
        if kind == "cw":
            return self.cw(x, meta, goal, steps=cw_steps)   # C&W picks its own size: eps is not used
        if kind == "off" or eps <= 0:
            return x
        if kind == "noise":
            return self.noise(x, meta, eps)
        if kind == "fgsm":
            return self.fgsm(x, meta, eps, goal)
        if kind == "pgd":
            return self.pgd(x, meta, eps, goal, steps)
        raise ValueError(f"Unknown attack: {kind}")
