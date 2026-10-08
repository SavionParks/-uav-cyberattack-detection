# Residual CNN Traffic-Sign Robustness Demo

This folder adds a **Stop / 35 mph / 45 mph traffic-sign experiment** to the UAV cyberattack-detection project.

It uses the same style of camera pipeline already used in the repository:

```text
Camera -> YOLO traffic-sign detector -> detected sign crop -> Residual CNN -> class prediction
```

The adversarial portion is an **offline FGSM robustness test on saved images only**. It does not apply adversarial perturbations to a live camera stream, vehicle, or drone control system.

## Files

- `Residual_CNN_Traffic_Signs_Colab.ipynb` - trains the TensorFlow/Keras Residual CNN on Stop, 35 mph, and 45 mph signs and evaluates offline FGSM robustness.
- `residual_cnn_camera.py` - uses the repository-style OpenCV + YOLO camera setup, then classifies each detected traffic-sign crop with the Residual CNN.
- `offline_fgsm_demo.py` - visualizes an original saved image, the FGSM perturbation, and the perturbed image.
- `download_demo_images.py` - downloads three openly reusable traffic-sign images for a quick demo.
- `requirements.txt` - local Python packages.

## 1. Train the Residual CNN in Google Colab

Open `Residual_CNN_Traffic_Signs_Colab.ipynb` in Google Colab and run it from top to bottom.

The notebook:

1. downloads the traffic-sign classification dataset,
2. keeps the Stop / 35 / 45 classes,
3. resizes images to `64 x 64 RGB`,
4. balances the training data with SMOTE,
5. trains the Residual CNN,
6. evaluates clean test accuracy,
7. evaluates FGSM on held-out test images.

At the end, download these two files from Colab:

```text
residual_cnn_traffic_signs.keras
residual_classes.json
```

Put them inside `traffic_sign_residual_demo/`.

## 2. Install the local requirements

From the repository root:

```bash
pip install -r traffic_sign_residual_demo/requirements.txt
```

## 3. Download demo sign images

From the repository root:

```bash
python traffic_sign_residual_demo/download_demo_images.py
```

It creates:

```text
traffic_sign_residual_demo/demo_images/stop.png
traffic_sign_residual_demo/demo_images/speed_limit_35.png
traffic_sign_residual_demo/demo_images/speed_limit_45.png
```

The source images are openly reusable Wikimedia Commons traffic-sign graphics:

- Stop sign: https://commons.wikimedia.org/wiki/File:MUTCD_Stop_Sign_(R1-1).svg - CC0/public-domain dedication.
- Speed Limit 35: https://commons.wikimedia.org/wiki/File:Speed_Limit_35_sign.svg - public domain.
- Speed Limit 45: https://commons.wikimedia.org/wiki/File:Speed_Limit_45_sign.svg - public domain.

These clean sign graphics are useful for a quick sanity check. For a stronger demo, also test photos captured from the same camera used in the live setup, because real photos include angle, lighting, distance, and background variation.

## 4. Run the live camera classifier

You still need the repository's trained YOLO traffic-sign `.pt` model.

Example:

```bash
python traffic_sign_residual_demo/residual_cnn_camera.py --weights best.pt --classifier traffic_sign_residual_demo/residual_cnn_traffic_signs.keras --classes traffic_sign_residual_demo/residual_classes.json
```

The default camera source is `0`.

For a USB camera:

```bash
python traffic_sign_residual_demo/residual_cnn_camera.py --weights best.pt --source 1
```

For a saved video:

```bash
python traffic_sign_residual_demo/residual_cnn_camera.py --weights best.pt --source drive.mp4
```

Press `Q` to quit.

The live display shows both predictions, for example:

```text
YOLO: stop_sign 0.94
Residual CNN: Stop 0.97
AGREE
```

`AGREE` means the YOLO sign label and Residual CNN label match. `CHECK` means they do not.

## 5. Show the offline FGSM attack

First download the sample images:

```bash
python traffic_sign_residual_demo/download_demo_images.py
```

Then move into the demo folder:

```bash
cd traffic_sign_residual_demo
```

Run the Stop sign test:

```bash
python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.03
```

Run the 35 mph test:

```bash
python offline_fgsm_demo.py --image demo_images/speed_limit_35.png --label speedLimit35 --epsilon 0.03
```

Run the 45 mph test:

```bash
python offline_fgsm_demo.py --image demo_images/speed_limit_45.png --label speedLimit45 --epsilon 0.03
```

The script displays:

```text
Original image | Visible FGSM perturbation | Perturbed image
```

and prints results similar to:

```text
True class: Stop
Before attack: Stop 97.42%
After attack: speedLimit45 71.18%
Epsilon: 0.03
ATTACK SUCCESSFUL: prediction changed
```

Your exact prediction and confidence will depend on the trained model and image.

The visualization is also saved as:

```text
fgsm_demo_result.png
```

## 6. Recommended presentation order

1. Run the live camera and show a normal Stop, 35 mph, or 45 mph detection.
2. Explain that YOLO locates the sign while the Residual CNN provides a second classification.
3. Save/capture a normal sign image from the camera.
4. Run that saved image through `offline_fgsm_demo.py`.
5. Show the original image, perturbation, and perturbed image side by side.
6. Compare the prediction/confidence before and after FGSM.
7. Repeat with several held-out images and report overall clean vs. attacked accuracy rather than only showing one successful example.

## 7. Try different FGSM strengths

For a controlled class experiment, compare small epsilon values on the held-out test set, for example:

```bash
python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.01
python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.03
python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.05
```

The Colab notebook is the better place to report aggregate accuracy across the entire test set.

## Notes

- The Residual CNN expects `64 x 64 RGB` images normalized to `[0, 1]`.
- Model files (`.keras`, `.pt`) are ignored by this folder's `.gitignore`; train/download them locally.
- Keep the FGSM portion offline on saved/test images. The live camera script performs normal inference only.

# Residual CNN Camera Setup Instructions

The camera system works by using YOLO to detect a traffic sign and then sending the detected sign crop to the Residual CNN for a second classification.

The process is:

Camera → YOLO Detection → Sign Crop → Residual CNN → Final Prediction

For example, when a stop sign is shown to the camera, the program may display:

```text
YOLO: stop_sign 0.94
Residual CNN: Stop 0.97
AGREE
```

`AGREE` means that both YOLO and the Residual CNN identified the same traffic sign.

## Required Files

Make sure the `traffic_sign_residual_demo` folder contains:

```text
traffic_sign_residual_demo/
├── residual_cnn_camera.py
├── residual_cnn_traffic_signs.keras
├── residual_classes.json
├── best.pt
└── requirements.txt
```

The files have the following purposes:

- `residual_cnn_camera.py` runs the live camera.
- `residual_cnn_traffic_signs.keras` is the trained Residual CNN model.
- `residual_classes.json` contains the Residual CNN class names.
- `best.pt` is the YOLO traffic-sign detection model.
- `requirements.txt` contains the Python packages needed to run the program.

## Step 1: Install the Requirements

Open Command Prompt or Terminal inside the main GitHub project folder.

Run:

```bash
pip install -r traffic_sign_residual_demo/requirements.txt
```

This installs TensorFlow, OpenCV, Ultralytics YOLO, NumPy, and the other packages needed for the project.

## Step 2: Run the Camera

From the main GitHub project folder, run:

```bash
python traffic_sign_residual_demo/residual_cnn_camera.py --weights traffic_sign_residual_demo/best.pt --classifier traffic_sign_residual_demo/residual_cnn_traffic_signs.keras --classes traffic_sign_residual_demo/residual_classes.json
```

The program will use the computer's default webcam.

The default camera source is:

```text
Camera 0
```

The requested camera resolution is:

```text
1280 x 720
```

## Step 3: Test a Traffic Sign

Hold or display one of the supported traffic signs in front of the camera.

The current classes are:

```text
Stop Sign
35 MPH Speed Limit
45 MPH Speed Limit
```

YOLO will first locate the sign and draw a box around it.

The detected sign is then cropped and sent to the Residual CNN.

The screen will display both model predictions.

Example:

```text
YOLO: 35_speed_limit 0.91
Residual CNN: speedLimit35 0.96
AGREE
```

If the models give different answers, the program will display:

```text
CHECK
```

This means YOLO and the Residual CNN did not agree on the sign classification.

## Using a USB Camera

If the built-in webcam is camera `0`, an external USB webcam may be camera `1`.

Run:

```bash
python traffic_sign_residual_demo/residual_cnn_camera.py --weights traffic_sign_residual_demo/best.pt --source 1
```

If camera `1` does not work, another camera number may need to be tested.

## Closing the Camera

Press:

```text
Q
```

while the camera window is selected to stop the program.

## Showing the FGSM Attack

The live camera is used for normal traffic-sign detection only.

The FGSM adversarial test is performed on a saved image instead of directly modifying the live camera feed.

The recommended demonstration is:

```text
1. Start the live camera.
2. Show a Stop, 35 MPH, or 45 MPH sign.
3. Show that YOLO and the Residual CNN recognize the sign.
4. Save or use an image of that traffic sign.
5. Run the saved image through the offline FGSM attack.
6. Compare the original prediction with the attacked prediction.
```

For example, after saving a stop-sign image, run:

```bash
cd traffic_sign_residual_demo
```

Then:

```bash
python offline_fgsm_demo.py --image demo_images/stop.png --label Stop --epsilon 0.03
```

The program will display:

```text
Original Image
      +
FGSM Perturbation
      =
Attacked Image
```

It will also print results similar to:

```text
True class: Stop
Before attack: Stop 97.42%
After attack: speedLimit45 71.18%
Epsilon: 0.03
ATTACK SUCCESSFUL: the offline prediction changed.
```

The exact results will depend on the trained model and the image being tested.

## Full Demonstration Flow

```text
LIVE CAMERA TEST

Camera
  ↓
YOLO Detects Sign
  ↓
Traffic Sign Crop
  ↓
Residual CNN
  ↓
Stop / 35 MPH / 45 MPH
  ↓
YOLO and CNN Comparison


OFFLINE ATTACK TEST

Saved Traffic Sign Image
  ↓
Normal Residual CNN Prediction
  ↓
FGSM Perturbation
  ↓
Residual CNN Prediction After Attack
  ↓
Compare Results
```

This setup allows the project to demonstrate both real-time traffic-sign recognition and an offline adversarial robustness test using the Residual CNN.
