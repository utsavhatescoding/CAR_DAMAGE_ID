# Nepal Vehicle Inspector — Three Models

This is the original Nepal Vehicle Inspector application and UI with one addition: our Colab-trained YOLOv8n is available as the third model. All original interface features remain unchanged.

## Install our trained model

Copy the validated 6.2 MB weight file to:

```text
models/cardd_yolov8n_detection_v1_best.pt
```

The application expects the six CarDD classes: dent, scratch, crack, glass shatter, lamp broken and tire flat.

## Models

- **YOLO11m — Precision:** original model option.
- **YOLOv8s — Fast:** original model option.
- **Our YOLOv8n — Colab trained:** our trained and independently tested model.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Push the project to GitHub, including the trained model at the exact path above, and deploy `app.py`. Confirm that your model and dataset permissions allow the weight file to be redistributed.

This application is AI-assisted visual screening only. Confidence is model certainty, not physical damage severity.
