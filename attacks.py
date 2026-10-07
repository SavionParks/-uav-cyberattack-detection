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
    and two goals:
      - "hide": make the detector miss signs
      - "swap": make a 45 mph sign read as 35 mph (or 35 as 45)

These are digital attacks on your own model, for robustness testing only.
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:  # older ultralytics versions
    from ultralytics.utils.ops import non_max_suppression

ATTACKS = ["off", "noise", "fgsm", "pgd"]
GOALS = ["hide", "swap"]


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
        self.names = self.yolo.names
        self.imgsz = int(imgsz)
        self.kind_to_idx = {sign_kind(v): k for k, v in self.names.items()}

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
        """Decide the swap direction from the clean image: 45 -> 35, or 35 -> 45 (None if no speed limit sign)."""
        i35, i45 = self.kind_to_idx.get("35"), self.kind_to_idx.get("45")
        if i35 is None or i45 is None:
            raise ValueError("Swap attack needs a model with both 35 and 45 mph classes.")
        with torch.no_grad():
            _, cls = self.class_scores(x)
        if max(cls[0, i45].max(), cls[0, i35].max()) < 0.25:
            return None  # no speed limit sign in view: nothing to swap
        src = i45 if cls[0, i45].max() >= cls[0, i35].max() else i35
        tgt = i35 if src == i45 else i45
        anchors = cls[0, src].topk(min(50, cls.shape[2])).indices
        return src, tgt, anchors

    def _loss(self, x, goal, pair=None):
        """A number the attack tries to make smaller."""
        _, cls = self.class_scores(x)
        cls = cls[0]  # (nc, anchors)
        if goal == "hide":
            # push down the strongest sign scores anywhere in the image
            return cls.flatten().topk(min(100, cls.numel())).values.sum()
        # "swap": lower the real speed limit's score and raise the other one, on the sign itself
        src, tgt, a = pair
        return (cls[src, a] - cls[tgt, a]).sum()

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

    def attack(self, kind, x, meta, eps, goal="hide", steps=10):
        if kind == "off" or eps <= 0:
            return x
        if kind == "noise":
            return self.noise(x, meta, eps)
        if kind == "fgsm":
            return self.fgsm(x, meta, eps, goal)
        if kind == "pgd":
            return self.pgd(x, meta, eps, goal, steps)
        raise ValueError(f"Unknown attack: {kind}")
