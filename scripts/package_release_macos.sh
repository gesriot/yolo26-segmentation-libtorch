#!/usr/bin/env bash
# Build a self-contained macOS release archive: the Rust binary, the bundled
# LibTorch runtime (CPU + MPS kernels), the yolo26s TorchScript model, COCO
# classes, a sample image and all licenses. The archive runs without a system
# PyTorch/LibTorch install.
#
#   scripts/package_release_macos.sh <vMAJOR.MINOR.PATCH> [output-dir]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:?usage: package_release_macos.sh <vX.Y.Z> [output-dir]}"
OUTPUT_DIR="${2:-${ROOT_DIR}/dist}"

if [[ ! "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.-]+)?$ ]]; then
    printf 'Invalid version: %s\n' "${VERSION}" >&2
    exit 2
fi

BINARY="${ROOT_DIR}/target/release/yolo26-seg"
TORCH_LIB="${TORCH_LIB_DIR:-${ROOT_DIR}/.venv/lib/python3.12/site-packages/torch/lib}"
MODEL="${ROOT_DIR}/models/yolo26s-seg.torchscript"
CLASSES="${ROOT_DIR}/shared/classes/coco80.txt"
SAMPLE="${ROOT_DIR}/shared/testdata/bus.jpg"
TORCH_LICENSE="${TORCH_LICENSE_FILE:-$(ls "${ROOT_DIR}"/.venv/lib/python3.12/site-packages/torch-*.dist-info/licenses/LICENSE 2>/dev/null | head -1)}"

for path in "${BINARY}" "${MODEL}" "${CLASSES}" "${SAMPLE}"; do
    if [[ ! -e "${path}" ]]; then
        printf 'Required artifact missing: %s\n' "${path}" >&2
        exit 1
    fi
done

# Runtime libraries the binary needs. libtorch_cpu carries the MPS kernels and
# pulls in libc10 and libomp; libtorch re-exports them. All inter-library
# references are @rpath, so co-locating them beside the binary is sufficient.
DYLIBS=(libtorch_cpu.dylib libtorch.dylib libc10.dylib libomp.dylib)

ASSET="yolo26-seg-${VERSION}-rust-macos"
STAGING="${OUTPUT_DIR}/.staging"
PKG="${STAGING}/${ASSET}"
ARCHIVE="${OUTPUT_DIR}/${ASSET}.zip"

rm -rf "${STAGING}"
mkdir -p "${PKG}/models" "${PKG}/classes" "${PKG}/sample" "${PKG}/licenses"

cp "${BINARY}" "${PKG}/yolo26-seg"
cp "${MODEL}" "${PKG}/models/yolo26s-seg.torchscript"
cp "${CLASSES}" "${PKG}/classes/coco80.txt"
cp "${SAMPLE}" "${PKG}/sample/bus.jpg"
cp "${ROOT_DIR}/LICENSE" "${PKG}/licenses/LICENSE"

for lib in "${DYLIBS[@]}"; do
    if [[ ! -f "${TORCH_LIB}/${lib}" ]]; then
        printf 'LibTorch runtime library not found: %s/%s\n' "${TORCH_LIB}" "${lib}" >&2
        exit 1
    fi
    cp "${TORCH_LIB}/${lib}" "${PKG}/${lib}"
done

if [[ -n "${TORCH_LICENSE}" && -f "${TORCH_LICENSE}" ]]; then
    cp "${TORCH_LICENSE}" "${PKG}/licenses/LibTorch-LICENSE.txt"
else
    printf 'warning: LibTorch license not found; archive will omit it\n' >&2
fi

# Ultralytics ships the YOLO26 weights under AGPL-3.0; include the matching text.
ULTRALYTICS_LICENSE_URL="https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.71/LICENSE"
if ! curl -fsSL "${ULTRALYTICS_LICENSE_URL}" -o "${PKG}/licenses/Ultralytics-AGPL-3.0.txt"; then
    printf 'Failed to download the Ultralytics AGPL-3.0 license\n' >&2
    exit 1
fi

# The binary resolves @rpath/libtorch*.dylib; make it look beside itself. The
# bundled libraries reference each other through the same run-path list.
chmod u+w "${PKG}/yolo26-seg"
if ! otool -l "${PKG}/yolo26-seg" |
    awk '/LC_RPATH/{getline;getline;print $2}' | grep -Fxq '@executable_path'; then
    install_name_tool -add_rpath '@executable_path' "${PKG}/yolo26-seg"
fi

# install_name_tool invalidates the ad-hoc signature; re-sign everything.
codesign --force --sign - "${PKG}"/*.dylib
codesign --force --sign - "${PKG}/yolo26-seg"

cat >"${PKG}/README.txt" <<EOF
YOLO26 segmentation ${VERSION} (Rust + LibTorch, macOS arm64)

Self-contained: bundles the LibTorch runtime (CPU and Metal/MPS), the YOLO26s
TorchScript model, the COCO class list and a sample image.

Example (Metal/MPS with automatic CPU fallback):
  ./yolo26-seg \\
    --model models/yolo26s-seg.torchscript \\
    --input sample/bus.jpg \\
    --classes classes/coco80.txt \\
    --output result.png \\
    --json result.json \\
    --device auto

The project source is MIT licensed (licenses/LICENSE). The bundled LibTorch
runtime is BSD licensed (licenses/LibTorch-LICENSE.txt). The YOLO26 model is
governed by the Ultralytics AGPL-3.0 license (licenses/Ultralytics-AGPL-3.0.txt).
EOF

(cd "${PKG}" && shasum -a 256 yolo26-seg models/yolo26s-seg.torchscript >SHA256SUMS.txt)

mkdir -p "${OUTPUT_DIR}"
rm -f "${ARCHIVE}"
ditto -c -k --keepParent --norsrc "${PKG}" "${ARCHIVE}"

# Verify the packaged archive end to end: extract it elsewhere and run the
# binary with a clean environment so only the bundled libraries are visible.
VERIFY="${STAGING}/verify"
rm -rf "${VERIFY}"
mkdir -p "${VERIFY}"
ditto -x -k "${ARCHIVE}" "${VERIFY}"
env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH \
    "${VERIFY}/${ASSET}/yolo26-seg" \
    --model "${VERIFY}/${ASSET}/models/yolo26s-seg.torchscript" \
    --input "${VERIFY}/${ASSET}/sample/bus.jpg" \
    --classes "${VERIFY}/${ASSET}/classes/coco80.txt" \
    --output "${VERIFY}/result.png" \
    --json "${VERIFY}/result.json" \
    --device auto --warmup 0 --iterations 1 >/dev/null
rm -rf "${STAGING}/.staging" "${VERIFY}"
printf 'Created %s\n' "${ARCHIVE}"