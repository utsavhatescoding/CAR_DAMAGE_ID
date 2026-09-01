# Nepal Vehicle Inspector — Model Comparison

This Streamlit application compares three published car-damage models while
preserving the original Nepal Vehicle Inspector interface. Checkpoints are
downloaded once and cached by the application.

## Models

- **YOLO11m — Precision:** original model option.
- **YOLOv8s — Fast:** original model option.
- **YOLO26m-seg — Best tested:** Cloudwhynot's segmentation checkpoint, which
  achieved the strongest result in our untouched CarDD test.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Push the project to GitHub and deploy `app.py`. The host requires outbound
internet access on first use so the published checkpoints can be downloaded.
Confirm that the relevant model and dataset licences permit your intended use.

This application is AI-assisted visual screening only. Confidence is model certainty, not physical damage severity.
