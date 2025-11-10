# Sign Language Recognition (Flask)

Real‑time sign language recognition web app built with Flask, OpenCV, MediaPipe, and TensorFlow/TFLite. It detects hand landmarks from your webcam and classifies them into letters/phrases, streaming results to a clean, responsive UI.

![App Screenshot](./image.png)

## Features

- Real‑time hand landmark detection (MediaPipe Hands)
- Two classifiers
  - Fast TFLite keypoint classifier for A–Z letters (`keypoint_classifier/*.tflite`)
  - Optional Keras/CNN phrase model (`Model/*.h5`), lazy‑loaded
- Flask server with MJPEG video streaming and live recognized text panel
- Polished UI with Home, About, Team, and Contact pages
- Pinned, deployment‑friendly dependencies for Windows (Python 3.10)

## Tech stack

- Backend: Python, Flask, OpenCV, MediaPipe, TensorFlow 2.12, TFLite
- Frontend: Bootstrap 5, Font Awesome, vanilla JS

## Project structure

```
app.py                       # Flask app (entrypoint)
requirements.txt             # Pinned runtime deps
keypoint_classifier/         # TFLite letter classifier and labels
Model/                       # Keras (.h5) models (optional/phrases)
static/                      # CSS, JS, images used by templates
templates/                   # Jinja2 templates (index, about, team, ...)
uploads/                     # User uploads (ignored by git)
images/                      # Local dataset samples (ignored by git)
```

Note: A nested folder `Sign-Language-Recognition/` is ignored to avoid duplicate content in the repo.

## Getting started (Windows, PowerShell)

Prerequisites:
- Windows 10/11
- Python 3.10.x (recommended for TensorFlow 2.12 wheels)
- A working webcam connected to the machine running the server

Setup:

```powershell
# 1) Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the app
python app.py
# Open http://127.0.0.1:5000 in your browser
```

Camera note: The server process reads frames from the local webcam (OpenCV). For cloud hosting, this approach won’t access a user’s browser camera; you’d use a client‑side capture (e.g., WebRTC) and send frames to the server or run inference in the browser.

## Usage tips

- On the Home page, use the UI buttons to toggle modes/classifiers if enabled.
- If you have multiple cameras, change the camera index in `app.py` (look for `cv2.VideoCapture(0)` and try `1`, `2`, ...).
- Models must be present:
  - `keypoint_classifier/keypoint_classifier.tflite` and its label CSV
  - `Model/sign_language_model_improved.h5` (optional)

## Troubleshooting

- TensorFlow import error or mismatched wheels
  - Use Python 3.10 and TensorFlow 2.12 as pinned in `requirements.txt`.
- Protobuf / MediaPipe errors
  - This project pins `protobuf==3.20.3` which is compatible with the selected stack.
- Camera not opening or busy
  - Close other apps using the webcam (Zoom, Teams, Camera app) and retry.
- High CPU usage
  - Use the TFLite keypoint classifier mode; reduce frame size or FPS if needed.
- Blank video feed in browser
  - Ensure the server console shows frames being processed and no OpenCV errors.

## Development

- Templates live in `templates/` and static assets in `static/`.
- Update team info in `templates/team.html` and `templates/about.html`.
- Large local datasets (`images/`) and user uploads (`uploads/`) are ignored by git.

## Team

- Sahil Gite — Lead Developer — GitHub: https://github.com/sahilgite1023 — LinkedIn: https://www.linkedin.com/in/sahilgite2003
- Anushka Shinde — ML Engineer — GitHub: https://github.com/anushkashinde7188 — Portfolio: https://anushkashinde.netlify.app/
- Shriram Mange — UI/UX Designer — GitHub: https://github.com/Shriram2005 — Portfolio: https://shrirammange.tech

## License

This repository is provided as‑is for educational and demonstrative purposes. Add a LICENSE file if you plan to distribute or open‑source under specific terms.
