# YOLO26 segmentation in Rust + LibTorch

Rust-only application code for YOLO26 instance segmentation. Inference uses
LibTorch through `tch`; image decoding, letterboxing, masks and rendering use
pure Rust crates. On Apple Silicon, `--device auto` tries Metal/MPS with FP16,
then MPS FP32, then CPU FP32.

The files in `models/*.pt` are Ultralytics Python checkpoints. LibTorch cannot
load that format directly, so they must be exported once to TorchScript.

## macOS setup

The `tch 0.24` crate targets LibTorch 2.11. The setup script installs the
matching official macOS PyTorch distribution into a project-local environment;
the compiled application itself is still Rust and does not embed Python.

```bash
scripts/setup-libtorch.sh
.venv/bin/python scripts/export_torchscript.py --device auto
scripts/cargo-libtorch.sh build --release
```

Export only one model while iterating:

```bash
.venv/bin/python scripts/export_torchscript.py models/yolo26n-seg.pt --device auto
```

## Run

```bash
scripts/cargo-libtorch.sh run --release -p yolo-seg-cli -- \
  --model models/yolo26n-seg.torchscript \
  --input /path/to/image.jpg \
  --classes shared/classes/coco80.txt \
  --output outputs/result.png \
  --json outputs/result.json \
  --device auto \
  --warmup 3 \
  --iterations 10
```

Useful performance switches:

- `--device auto|mps|cpu` selects the accelerator;
- `--precision auto|f16|f32` defaults to FP16 on MPS and FP32 on CPU;
- `--threads N` controls CPU inference workers;
- `--warmup` excludes graph/kernel warm-up from the reported average;
- `--iterations` reports stable average stage timings.

Use explicit `--device mps` to treat an MPS failure as an error. `auto` is the
production-safe mode with transparent fallback.

The exporter normalizes traced float64 constants that Metal cannot allocate.
The resulting artifact works on both CPU and MPS; `--device` only selects where
the one-time export computation runs. Loading maps the complete TorchScript
archive, including YOLO anchor constants, onto the selected inference device.

## Test

```bash
scripts/cargo-libtorch.sh test --workspace
```

The model checkpoints are governed by the Ultralytics license. The project
source is MIT licensed.

## Verified performance

Apple M4, `yolo26n-seg`, 640×640, three warmups and ten measured iterations:

| Backend | Preprocess | Inference | Postprocess | Total |
| --- | ---: | ---: | ---: | ---: |
| MPS FP16 | 7.18 ms | 50.87 ms | 8.24 ms | 66.29 ms |
| CPU FP32 | 14.70 ms | 412.06 ms | 8.67 ms | 435.43 ms |

That run was about 8.1× faster for model inference and 6.6× faster end to end
on MPS. All five supplied `n/s/m/l/x` models were smoke-tested on MPS FP16.
