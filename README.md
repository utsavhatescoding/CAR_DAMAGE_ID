# Nepal Vehicle Inspector

This Streamlit application performs car-only visual damage screening using a
two-stage segmentation pipeline. It preserves the original Nepal Vehicle
Inspector interface while supporting both image upload and mobile camera input.

## Inference pipeline

1. A pretrained YOLO26m-seg model finds car instances.
2. The largest car is selected as the inspection target.
3. A padded crop preserves normal image pixels and exterior context.
4. Cloudwhynot's YOLO26m-seg damage checkpoint runs at 896 pixels.
5. Damage masks must overlap the selected, slightly dilated vehicle silhouette.
6. Accepted masks are mapped back onto the original image.

The previous YOLO11m and YOLOv8s comparison options have been removed.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Push the project to GitHub and deploy `app.py`. The host requires outbound
internet access on first use so the two published checkpoints can be downloaded.
Confirm that the relevant model and dataset licences permit your intended use.

This application is AI-assisted visual screening only. Confidence is model certainty, not physical damage severity.
