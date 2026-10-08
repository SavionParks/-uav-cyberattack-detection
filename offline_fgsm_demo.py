"""Offline FGSM robustness demo for the trained Residual CNN.

This script only perturbs a saved image. It does not modify a live camera feed.

Example:
    python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.03
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 64


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def fgsm_attack(model, image, label_index, epsilon):
    x = tf.convert_to_tensor(image, dtype=tf.float32)
    y = tf.convert_to_tensor([label_index], dtype=tf.int64)

    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x, training=False)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)

    gradient = tape.gradient(loss, x)
    attacked = x + epsilon * tf.sign(gradient)
    return tf.clip_by_value(attacked, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Offline FGSM demo for the Residual CNN traffic-sign model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", required=True, help="Stop, speedLimit35, or speedLimit45")
    parser.add_argument("--model", default="residual_cnn_traffic_signs.keras")
    parser.add_argument("--classes", default="residual_classes.json")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--output", default="fgsm_demo_result.png")
    args = parser.parse_args()

    for p in [args.image, args.model, args.classes]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing file: {p}")

    with open(args.classes, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if args.label not in class_names:
        raise ValueError(f"Unknown label {args.label!r}. Choose one of {class_names}")

    true_index = class_names.index(args.label)
    model = tf.keras.models.load_model(args.model)
    clean = load_image(args.image)

    clean_probs = model.predict(clean, verbose=0)[0]
    clean_index = int(np.argmax(clean_probs))
    clean_conf = float(clean_probs[clean_index])

    attacked = fgsm_attack(model, clean, true_index, args.epsilon)
    attacked_np = attacked.numpy()
    attack_probs = model.predict(attacked_np, verbose=0)[0]
    attack_index = int(np.argmax(attack_probs))
    attack_conf = float(attack_probs[attack_index])

    perturbation = attacked_np[0] - clean[0]
    if args.epsilon > 0:
        visible = perturbation / (2.0 * args.epsilon) + 0.5
    else:
        visible = np.full_like(perturbation, 0.5)
    visible = np.clip(visible, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(clean[0])
    axes[0].set_title(f"Original\n{class_names[clean_index]} {clean_conf*100:.1f}%")
    axes[1].imshow(visible)
    axes[1].set_title(f"FGSM perturbation\nepsilon={args.epsilon}")
    axes[2].imshow(attacked_np[0])
    axes[2].set_title(f"After FGSM\n{class_names[attack_index]} {attack_conf*100:.1f}%")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.show()

    print("True class:", args.label)
    print("Before attack:", class_names[clean_index], f"{clean_conf*100:.2f}%")
    print("After attack:", class_names[attack_index], f"{attack_conf*100:.2f}%")
    print("Epsilon:", args.epsilon)
    print("ATTACK SUCCESSFUL: prediction changed" if clean_index != attack_index else "ATTACK UNSUCCESSFUL: prediction stayed the same")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
