Optional local RF-DETR placement:

`models/checkpoint_best_ema.pth`

Cloudwhynot's damage checkpoint and the vehicle-segmentation checkpoint are
downloaded from their public repository and cached when first used. On hosted
deployments, configure our RF-DETR checkpoint through `RFDETR_MODEL_URL` rather
than committing the 137 MB file as an ordinary GitHub blob.
