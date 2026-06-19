# Models

Place the original Ultralytics checkpoints here as `yolo26*-seg.pt`, then run:

```bash
.venv/bin/python scripts/export_torchscript.py --device auto
```

The generated `*.torchscript` files are ready for both LibTorch CPU and
Metal/MPS inference. Model binaries are intentionally ignored by Git.

