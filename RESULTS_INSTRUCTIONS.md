# Residual CNN — 80% Accuracy Target and FGSM Results

This update adds an **offline traffic-sign classification and adversarial robustness experiment** to the existing UAV repository. The experiment uses the same 64×64 TensorFlow/Keras Residual CNN architecture, with three sign classes: `Stop`, `speedLimit35`, and `speedLimit45`.

**The 80% values are goals, not measured results.** You must run the notebook to obtain honest performance results.

## 1. Run the new Colab notebook

Open **`Residual_CNN_80pct_Results_Colab.ipynb`** from this repository in [Google Colab](https://colab.research.google.com/). Choose **Runtime → Run all**.

The notebook downloads a traffic-sign dataset, trains a Residual CNN, and evaluates it on held-out images. It aims for **80% clean classification accuracy** and tests whether FGSM reaches **80% attack success** on images originally classified correctly. It chooses the model and FGSM epsilon on validation data before evaluating the held-out test set. It does not fabricate an 80% result when that target is missed.

Once the notebook runs, download:

```text
residual_cnn_traffic_signs.keras
residual_classes.json
traffic_sign_measured_results.zip
```

`traffic_sign_measured_results.zip` contains the actual measured results, including:

- `metrics.json` — clean accuracy, accuracy after FGSM, attack success rate, selected epsilon, and goal status.
- `fgsm_test_results.csv` and `fgsm_validation_results.csv` — per-epsilon results.
- `fgsm_results_chart.png` — accuracy and attack success graph.
- `confusion_matrix_clean.png` and `confusion_matrix_fgsm.png` — classification confusion matrices.
- `fgsm_example.png` — clean / perturbation / adversarial comparison.

**Clean accuracy** is the share of all held-out images classified correctly. **Attack success rate** is the share of initially correct images that become misclassified after FGSM; it is not classifier accuracy.

## 2. Install packages for local camera/demo use

Clone or download the repository, open a terminal in its root folder, and run:

```bash
python -m pip install -r requirements.txt
```

Put `residual_cnn_traffic_signs.keras` and `residual_classes.json` in the repository root, next to `residual_cnn_camera.py`. The repo already includes the YOLO checkpoint `Detecting stop signs traffic (3).pt`.

## 3. Run normal live traffic-sign detection

Run this from the repository root (Windows Command Prompt/PowerShell or any terminal):

```bash
python residual_cnn_camera.py --weights "Detecting stop signs traffic (3).pt" --classifier residual_cnn_traffic_signs.keras --classes residual_classes.json
```

For an external camera, add `--source 1`. The default webcam is `0`. Press **Q** to exit.

YOLO draws bounding boxes for supported signs. The Residual CNN then classifies each detected crop and displays `AGREE` if both class labels match, or `CHECK` if they differ. **The live camera does not run FGSM.**

## 4. Download sign images and demonstrate FGSM offline

Run from the repo root:

```bash
python download_demo_images.py
```

This downloads illustrative Stop, 35 mph, and 45 mph sign graphics into `demo_images/` (see `PHOTO_SOURCES.md` for the image sources). Then run:

```bash
python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.03
```

Or replace the image and label with `demo_images/speed_limit_35.png` / `speedLimit35`, or `demo_images/speed_limit_45.png` / `speedLimit45`.

The script displays and saves `fgsm_demo_result.png` showing **Original → FGSM perturbation → Attacked image**, and reports whether the offline prediction changed. This one-image illustration is not the dataset-wide accuracy measurement.

## 5. Report the real results

After running Colab, open the downloaded `metrics.json` and report:

> Our Residual CNN reached **[test_clean_accuracy × 100]%** clean held-out accuracy. With FGSM epsilon **[selected_epsilon_validation_only]**, its accuracy after perturbation was **[test_accuracy_after_fgsm × 100]%**. The FGSM attack success rate among originally correct predictions was **[test_attack_success_rate × 100]%**.

Use `fgsm_results_chart.png` and `fgsm_example.png` in the presentation. If either 80% goal is missed, report that honestly. Digital FGSM on saved images is an offline robustness evaluation, not evidence of defeating a real vehicle/drone camera system.
