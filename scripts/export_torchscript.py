#!/usr/bin/env python3
"""Export local Ultralytics YOLO26 segmentation checkpoints for LibTorch."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))

import torch
from ultralytics import YOLO


def tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def normalize_mps_constants(path: Path) -> int:
    """Replace traced float64 constants, which the MPS backend cannot allocate."""
    module = torch.jit.load(path, map_location="cpu").eval()
    converted = 0

    def visit_block(block: Any) -> None:
        nonlocal converted
        for node in block.nodes():
            for attribute in node.attributeNames():
                if node.kindOf(attribute) != "t":
                    continue
                value = node.t(attribute)
                if value.dtype == torch.float64:
                    node.t_(attribute, value.float())
                    converted += 1
            for nested in node.blocks():
                visit_block(nested)

    for child in module.modules():
        for method_name in child._c._method_names():
            visit_block(child._c._get_method(method_name).graph)
    if converted:
        torch.jit.save(module, path)
    return converted


def validate(path: Path, image_size: int) -> list[tuple[int, ...]]:
    module = torch.jit.load(path, map_location="cpu").eval()
    with torch.inference_mode():
        output = module(torch.zeros(1, 3, image_size, image_size))
    shapes = [tuple(tensor.shape) for tensor in tensors(output)]
    prototypes = next((shape for shape in shapes if len(shape) == 4 and shape[0] == 1), None)
    if prototypes is None:
        raise RuntimeError(f"No [1,C,H,W] prototype tensor in {shapes}")
    channels = prototypes[1]
    detections = next(
        (
            shape
            for shape in shapes
            if len(shape) == 3
            and shape[0] == 1
            and (shape[1] == channels + 6 or shape[2] == channels + 6)
        ),
        None,
    )
    if detections is None:
        raise RuntimeError(f"No YOLO26 [1,N,{channels + 6}] detection tensor in {shapes}")
    return shapes


def export(
    checkpoint: Path,
    output_dir: Path,
    image_size: int,
    force: bool,
    device: str,
) -> Path:
    destination = output_dir / f"{checkpoint.stem}.torchscript"
    if destination.is_file() and not force:
        converted = normalize_mps_constants(destination)
        shapes = validate(destination, image_size)
        print(
            f"Keeping existing {destination} ({shapes}); "
            f"normalized {converted} float64 constant(s)"
        )
        return destination

    model = YOLO(checkpoint)
    exported = Path(
        model.export(
            format="torchscript",
            imgsz=image_size,
            batch=1,
            optimize=False,
            half=False,
            device=device,
        )
    ).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exported != destination.resolve():
        shutil.copy2(exported, destination)
    converted = normalize_mps_constants(destination)
    shapes = validate(destination, image_size)
    print(
        f"Exported {checkpoint.name} -> {destination} ({shapes}); "
        f"normalized {converted} float64 constant(s)"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path, nargs="*", help="checkpoint .pt files")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="export device; auto uses MPS when available",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    output_dir = (args.output_dir or model_dir).resolve()
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoints = [path.resolve() for path in args.models]
    if not checkpoints:
        checkpoints = sorted(model_dir.glob("yolo26*-seg.pt"))
    if not checkpoints:
        parser.error("no YOLO26 segmentation checkpoints found")
    for checkpoint in checkpoints:
        export(checkpoint, output_dir, args.imgsz, args.force, device)


if __name__ == "__main__":
    main()
