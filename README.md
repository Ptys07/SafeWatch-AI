# 🛡️ SafeWatch AI - Real-Time Emergency Detection System

**SafeWatch AI** is an AI-powered surveillance system designed to **automatically detect emergencies** in real-time CCTV feeds — such as **fires**, **accidents**, **crimes**, and **falls** — and place a **voice call alert** to emergency services like 911. This is intended to help communities get faster assistance during critical events without waiting for a human to react.

> ⚠️ **Disclaimer:** This is a prototype intended for educational or proof-of-concept use only. Always comply with local laws and ethical guidelines before deploying any AI surveillance or auto-call system in production.

---

## 🎯 Features

- 🔥 **Fire Detection** – Detects flames or smoke via HSV filtering.
- 🧍 **Fall Detection** – Uses MediaPipe pose estimation to detect sudden body collapse.
- 👮 **Crime Detection** – Detects weapons (guns/knives) using YOLOv5 object detection.
- 🚗 **Accident Detection** – Detects vehicle collisions based on overlapping bounding boxes.
- ☎️ **Auto Emergency Calling** – Places a voice call to 911 (or test number) via [SignalWire](https://signalwire.com).
- 🎥 **Live Video Feed** – Annotated display using OpenCV.

---

## 🧠 How It Works

1. The script captures live video from your webcam.
2. It continuously analyzes each frame using YOLOv5 and MediaPipe Pose.
3. If an emergency is detected:
   - It generates a human-like emergency alert message.
   - It places a phone call using the SignalWire API.
   - It delivers the alert message using TwiML via a local Flask server.

---

## 🚀 Getting Started

### 📦 Requirements

Install the required Python packages:

```bash
pip install opencv-python flask torch torchvision torchaudio requests mediapipe
