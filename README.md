# Nepal Vehicle Inspector

A mobile-first Streamlit application for visually testing two instance-segmentation
damage models on real vehicle photographs.

## Available models

### Our RF-DETR Seg Medium

- Our CarDD-trained `checkpoint_best_ema.pth`
- 624-pixel inference resolution
- Direct inference on the complete original image
- Native RF-DETR boxes, masks, classes and confidence scores
- No vehicle crop or overlap post-filter

### Cloudwhynot YOLO26

- Cloudwhynot's YOLO26m-seg damage checkpoint
- 896-pixel damage inference
- Largest-car silhouette isolation
- Vehicle-overlap filtering before results are displayed

These are intentionally labelled as different pipelines. Their displayed outputs
must not be interpreted as a controlled model benchmark unless preprocessing,
thresholds and evaluation data are standardized separately.

## Configure our RF-DETR checkpoint

The 137 MB RF-DETR checkpoint should not be committed as an ordinary GitHub file.
Host it at a permanent direct-download URL, then add this Streamlit secret:

```toml
RFDETR_MODEL_URL = "https://your-permanent-direct-download/checkpoint_best_ema.pth"
```

For testing, this may also be a normal Google Drive file-sharing link. Set the
file's General access to **Anyone with the link**. The application uses `gdown`
for Drive's large-file confirmation flow and validates the downloaded size.

For local development, either define the same value as an environment variable or
place the checkpoint at:

```text
models/checkpoint_best_ema.pth
```

The application validates that the downloaded file is at least 120 MB, downloads
to a temporary `.part` file, and only promotes a complete download into the cache.

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open the app in Streamlit Community Cloud.
3. Select **Manage app → Settings → Secrets**.
4. Add `RFDETR_MODEL_URL` using the TOML example above.
5. Reboot the app.

RF-DETR Medium is substantially heavier than YOLO and CPU inference can be slow on
free hosting. The application therefore loads models only when selected and caches
them after first use.

## Important limitation

This is an experimental AI-assisted visual screen, not a safety inspection,
repair estimate or insurance assessment. The CarDD models have shown domain shift
and false positives on clean real-world vehicles; every finding requires human
review.
