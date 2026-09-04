#!/usr/bin/env bash
set -euo pipefail

OPENCV_VERSION="${NPEC_OPENCV_VERSION:-4.11.0.86}"
DEPLOYMENT_TARGET="${NPEC_MACOS_DEPLOYMENT_TARGET:-13.0}"
PYTHON_BIN="${NPEC_BUILD_PYTHON:-$(command -v python3.11 || command -v python3)}"
OUTPUT_DIR="${1:-$(pwd)/dist-opencv}"
BUILD_ENV="$(mktemp -d /tmp/npec-opencv-build.XXXXXX)"

cleanup() {
  rm -rf "${BUILD_ENV}"
}
trap cleanup EXIT

mkdir -p "${OUTPUT_DIR}"
"${PYTHON_BIN}" -m venv "${BUILD_ENV}/venv"
"${BUILD_ENV}/venv/bin/pip" install --upgrade pip

export MACOSX_DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET}"
export CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=${DEPLOYMENT_TARGET} -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF"
"${BUILD_ENV}/venv/bin/pip" wheel \
  --no-deps \
  --no-cache-dir \
  --no-binary=opencv-python-headless \
  "opencv-python-headless==${OPENCV_VERSION}" \
  --wheel-dir "${OUTPUT_DIR}"

WHEEL_PATH="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "opencv_python_headless-${OPENCV_VERSION}-*.whl" -print -quit)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "OpenCV wheel was not produced."
  exit 1
fi
if unzip -l "${WHEEL_PATH}" | grep -Eiq '(libx264|libx265|libavcodec|libavformat)'; then
  echo "OpenCV wheel unexpectedly contains video-codec libraries: ${WHEEL_PATH}"
  exit 1
fi

WHEEL_DIR="$(dirname "${WHEEL_PATH}")"
WHEEL_NAME="$(basename "${WHEEL_PATH}")"
(
  cd "${WHEEL_DIR}"
  shasum -a 256 "${WHEEL_NAME}" > "${WHEEL_NAME}.sha256"
)
echo "Built no-FFmpeg OpenCV wheel: ${WHEEL_PATH}"
