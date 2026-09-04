#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_VENV="${NPEC_BUILD_VENV:-${SCRIPT_DIR}/.build-venv}"
REUSE_BUILD_VENV="${NPEC_REUSE_BUILD_VENV:-0}"
OPENCV_WHEEL="${NPEC_OPENCV_WHEEL:-}"
DIST_DIR="${NPEC_DIST_DIR:-${SCRIPT_DIR}/dist}"
BUILD_DIR="${NPEC_BUILD_DIR:-${SCRIPT_DIR}/build}"
PYTHON_BIN="${NPEC_BUILD_PYTHON:-}"
APP_NAME="NPEC Labeling Tool"
BUNDLE_ID="${NPEC_BUNDLE_ID:-nl.npec.labelingtool}"
APP_VERSION="${NPEC_APP_VERSION:-2026.8.2.5}"
MIN_MACOS_VERSION="${NPEC_MIN_MACOS_VERSION:-14.0}"
BUNDLE_FFMPEG="${NPEC_BUNDLE_FFMPEG:-0}"
MODEL_PROFILE="${NPEC_MODEL_PROFILE:-none}"
REQUIREMENTS_FILE="${NPEC_REQUIREMENTS_FILE:-${ROOT_DIR}/environments/macos-arm64-${APP_VERSION}.txt}"
export PYINSTALLER_CONFIG_DIR="${SCRIPT_DIR}/.pyinstaller-cache"

# PyInstaller's Mach-O rewriting must use Apple's Xcode toolchain on macOS.
# Conda's cctools-port install_name_tool is present earlier in PATH here and
# has been causing nondeterministic COLLECT failures during packaging.
XCODE_TOOLCHAIN_BIN="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin"
if [[ -d "${XCODE_TOOLCHAIN_BIN}" ]]; then
  export PATH="${XCODE_TOOLCHAIN_BIN}:/usr/bin:/bin:/usr/sbin:/sbin:${PATH}"
fi

safe_rmtree() {
  local target="$1"
  if [[ -e "${target}" ]]; then
    "${PYTHON_BIN:-python3}" - "${target}" <<'PY'
import shutil
import sys
from pathlib import Path
shutil.rmtree(Path(sys.argv[1]), ignore_errors=False)
PY
  fi
}

safe_rmtree "${PYINSTALLER_CONFIG_DIR}"
mkdir -p "${PYINSTALLER_CONFIG_DIR}"

if [[ -n "${PYTHON_BIN}" ]]; then
  :
elif [[ "${REUSE_BUILD_VENV}" == "1" && -x "${BUILD_VENV}/bin/python" ]]; then
  PYTHON_BIN="${BUILD_VENV}/bin/python"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
else
  PYTHON_BIN="$(command -v python3)"
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        "NPEC Labeling Tool requires Python >= 3.10 for packaging. "
        f"Selected interpreter is {sys.version.split()[0]} at {sys.executable}. "
        "Set NPEC_BUILD_PYTHON to a Python 3.10+ executable."
    )
PY

if [[ "${REUSE_BUILD_VENV}" == "1" ]]; then
  if [[ ! -x "${BUILD_VENV}/bin/python" ]] || [[ ! -x "${BUILD_VENV}/bin/pyinstaller" ]]; then
    echo "Reusable build environment is incomplete: ${BUILD_VENV}"
    exit 1
  fi
  echo "[1/5] Reusing build virtual environment: ${BUILD_VENV}"
  echo "[2/5] Build dependencies already installed."
else
  echo "[1/5] Creating build virtual environment with ${PYTHON_BIN}..."
  safe_rmtree "${BUILD_VENV}"
  "${PYTHON_BIN}" -m venv "${BUILD_VENV}"

  echo "[2/5] Installing build dependencies..."
  "${BUILD_VENV}/bin/pip" install --upgrade \
    "pip==26.2.1" \
    "setuptools==84.0.0" \
    "wheel==0.48.0"
  if [[ -n "${OPENCV_WHEEL}" ]]; then
    "${BUILD_VENV}/bin/pip" install "${OPENCV_WHEEL}"
  fi
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Release lock file not found: ${REQUIREMENTS_FILE}"
    exit 1
  fi
  "${BUILD_VENV}/bin/pip" install -r "${REQUIREMENTS_FILE}"
fi

ICON_SCRIPT="${SCRIPT_DIR}/scripts/generate_app_icons.py"
ICON_SOURCE="${SCRIPT_DIR}/assets/npec-logo-label.jpg"
ICON_ICNS="${SCRIPT_DIR}/assets/npec-app-icon.icns"
ICON_ARG=()
if [[ -f "${ICON_SCRIPT}" && -f "${ICON_SOURCE}" ]]; then
  echo "[2.5/5] Generating app icon assets..."
  "${BUILD_VENV}/bin/python" "${ICON_SCRIPT}" --input "${ICON_SOURCE}" --output-dir "${SCRIPT_DIR}/assets" || true
fi
if [[ -f "${ICON_ICNS}" ]]; then
  ICON_ARG=(--icon "${ICON_ICNS}")
fi

OCR_HELPER_SOURCE="${SCRIPT_DIR}/scripts/macos_vision_plate_ocr.swift"
OCR_HELPER_DIR="${SCRIPT_DIR}/bin"
OCR_HELPER_BIN="${OCR_HELPER_DIR}/npec_plate_ocr"
OCR_HELPER_ARG=()
if [[ -f "${OCR_HELPER_SOURCE}" ]] && command -v xcrun >/dev/null 2>&1; then
  echo "[2.6/5] Compiling Apple Vision plate OCR/barcode helper..."
  mkdir -p "${OCR_HELPER_DIR}"
  xcrun swiftc -O "${OCR_HELPER_SOURCE}" -o "${OCR_HELPER_BIN}"
  chmod +x "${OCR_HELPER_BIN}"
  OCR_HELPER_ARG=(--add-binary "${OCR_HELPER_BIN}:resources/bin")
fi

LEGAL_ARGS=()
for legal_file in LICENSE CITATION.cff AUTHORS.md THIRD_PARTY_NOTICES.md TRADEMARKS.md; do
  if [[ -f "${ROOT_DIR}/${legal_file}" ]]; then
    LEGAL_ARGS+=(--add-data "${ROOT_DIR}/${legal_file}:legal")
  fi
done
if [[ -d "${ROOT_DIR}/licenses" ]]; then
  LEGAL_ARGS+=(--add-data "${ROOT_DIR}/licenses:legal/licenses")
fi

MODEL_DATA_ARGS=()
if [[ "${MODEL_PROFILE}" == "yang" ]]; then
  YANG_MODEL_SPECS=(
    "2034ddeb36a980e2ebdb67871c059c2746ed7fbaba25e4e500d61bcb5a14059e|hades_lucifer/best_root_model_patch_256_max_f10.8_max_IoU0.905.h5"
    "5064497983311af53bc5a070f165fcf8ccb909226e42100d753c3ff059c962ea|rgb_inoculated/rgb_inoculated_shoot_v3.keras"
    "db54b5bf54ecccc0e79674351cf61dd00f7c8746abdf02ee1ad24f1b4931fe83|rgb_inoculated/rgb_inoculated_shoot_v3.profile.json"
    "a000451852dfd3ea088abeddcf62799e495efd67533a6ac9032890f4779bcb85|five_seedling_ownership/yang_rgb_owner_ranker_v1.json"
  )
  for model_spec in "${YANG_MODEL_SPECS[@]}"; do
    expected_sha256="${model_spec%%|*}"
    relative_path="${model_spec#*|}"
    model_path="${SCRIPT_DIR}/builtin_models/${relative_path}"
    if [[ ! -f "${model_path}" ]]; then
      echo "Yang Song workflow model profile is missing: ${model_path}"
      exit 1
    fi
    actual_sha256="$(shasum -a 256 "${model_path}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
      echo "Yang Song workflow model checksum mismatch: ${relative_path}"
      echo "Expected: ${expected_sha256}"
      echo "Actual:   ${actual_sha256}"
      exit 1
    fi
    model_family="${relative_path%%/*}"
    MODEL_DATA_ARGS+=(--add-data "${model_path}:resources/builtin_models/${model_family}")
  done
elif [[ "${MODEL_PROFILE}" == "none" ]]; then
  echo "Building without bundled checkpoint binaries."
elif [[ "${MODEL_PROFILE}" == "all" ]]; then
  for model_family in \
    hades_lucifer \
    potato \
    mxlab_dark_rgb \
    rgb_inoculated \
    five_seedling_ownership \
    general_root_starter_unified \
    plant_health; do
    model_path="${SCRIPT_DIR}/builtin_models/${model_family}"
    if [[ -d "${model_path}" ]]; then
      MODEL_DATA_ARGS+=(--add-data "${model_path}:resources/builtin_models/${model_family}")
    fi
  done
else
  echo "Unknown NPEC_MODEL_PROFILE: ${MODEL_PROFILE}. Expected all, yang, or none."
  exit 1
fi

# External pyphenotyper code is imported dynamically at runtime and expects
# these modules to exist inside the packaged app.
RUNTIME_COLLECTION_ARGS=(
  --hidden-import imageio
  --hidden-import imageio.v2
  --hidden-import imageio.v3
  --hidden-import imageio_ffmpeg
  --collect-all imageio
  --collect-all imageio_ffmpeg
  --copy-metadata imageio
  --copy-metadata imageio_ffmpeg
  --hidden-import patchify
  --collect-all patchify
  --hidden-import skimage
  --collect-all skimage
  --copy-metadata scikit-image
  --hidden-import tensorflow.lite
  --hidden-import tensorflow.lite.python.lite
  --hidden-import tensorflow.lite.python.convert
  --hidden-import tensorflow.lite.python.util
  --hidden-import tensorflow.lite.python.interpreter
  --hidden-import tensorflow.lite.python.schema_py_generated
  --hidden-import tensorflow.lite.python.authoring.authoring
  --hidden-import rich.progress
  --hidden-import zarr
  --collect-all rich
  --collect-all zarr
  --copy-metadata zarr
  --hidden-import openpyxl
  --collect-all openpyxl
  --hidden-import psd_tools
  --collect-all psd_tools
)

echo "[3/5] Running PyInstaller..."
safe_rmtree "${DIST_DIR}"
safe_rmtree "${BUILD_DIR}"
"${BUILD_VENV}/bin/pyinstaller" \
  --noconfirm \
  --windowed \
  --name "${APP_NAME}" \
  --osx-bundle-identifier "${BUNDLE_ID}" \
  --clean \
  ${ICON_ARG[@]+"${ICON_ARG[@]}"} \
  --add-data "${SCRIPT_DIR}/assets/npec-logo-label.jpg:resources/assets" \
  --add-data "${SCRIPT_DIR}/assets/npec-app-icon.png:resources/assets" \
  --add-data "${SCRIPT_DIR}/assets/NPEC-logo-black.png:resources/assets" \
  --add-data "${SCRIPT_DIR}/assets/npec-label.jpeg:resources/assets" \
  --add-data "${SCRIPT_DIR}/foundation_external_runner.py:resources" \
  --add-data "${SCRIPT_DIR}/plant_health_runner.py:resources" \
  --add-data "${SCRIPT_DIR}/npec_pyphenotyper:resources/npec_pyphenotyper" \
  --add-data "${SCRIPT_DIR}/general_root_starter:resources/general_root_starter" \
  ${LEGAL_ARGS[@]+"${LEGAL_ARGS[@]}"} \
  ${MODEL_DATA_ARGS[@]+"${MODEL_DATA_ARGS[@]}"} \
  ${OCR_HELPER_ARG[@]+"${OCR_HELPER_ARG[@]}"} \
  "${RUNTIME_COLLECTION_ARGS[@]}" \
  --distpath "${DIST_DIR}" \
  --workpath "${BUILD_DIR}" \
  --specpath "${SCRIPT_DIR}" \
  --paths "${ROOT_DIR}" \
  "${SCRIPT_DIR}/main.py"

APP_PATH="${DIST_DIR}/${APP_NAME}.app"
INFO_PLIST="${APP_PATH}/Contents/Info.plist"
if [[ -f "${INFO_PLIST}" ]]; then
  "${BUILD_VENV}/bin/python" - "${INFO_PLIST}" "${BUNDLE_ID}" "${APP_VERSION}" "${MIN_MACOS_VERSION}" <<'PY'
import plistlib
import sys
from pathlib import Path

info_plist = Path(sys.argv[1])
bundle_id = sys.argv[2]
app_version = sys.argv[3]
minimum_macos_version = sys.argv[4]

data = plistlib.loads(info_plist.read_bytes())
data["CFBundleIdentifier"] = bundle_id
data["CFBundleShortVersionString"] = app_version
data["CFBundleVersion"] = app_version
data["LSMinimumSystemVersion"] = minimum_macos_version
info_plist.write_bytes(plistlib.dumps(data, sort_keys=False))
PY
fi
if [[ "${BUNDLE_FFMPEG}" != "1" ]]; then
  echo "Removing imageio-ffmpeg executable from the public app bundle..."
  find "${APP_PATH}" -path '*imageio_ffmpeg/binaries/ffmpeg-*' \( -type f -o -type l \) -print -delete
  GPL_VIDEO_PATH="$(find "${APP_PATH}" -type f \( -name 'libx264*' -o -name 'libx265*' \) -print -quit)"
  if [[ -n "${GPL_VIDEO_PATH}" ]]; then
    echo "Public build refused: GPL video library remains in the app bundle: ${GPL_VIDEO_PATH}"
    echo "Use an OpenCV build without FFmpeg/x264/x265, or set NPEC_BUNDLE_FFMPEG=1 and meet all redistribution obligations."
    exit 1
  fi
fi
if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - --timestamp=none "${APP_PATH}"
fi
echo "Built app bundle: ${APP_PATH}"

if command -v create-dmg >/dev/null 2>&1; then
  echo "[4/5] Creating DMG installer..."
  DMG_PATH="${DIST_DIR}/${APP_NAME}.dmg"
  rm -f "${DMG_PATH}"
  create-dmg \
    --volname "${APP_NAME}" \
    --window-pos 200 120 \
    --window-size 800 420 \
    --icon-size 96 \
    --icon "${APP_NAME}.app" 180 190 \
    --app-drop-link 620 190 \
    "${DMG_PATH}" \
    "${APP_PATH}"
  echo "Built installer: ${DMG_PATH}"
else
  echo "[4/5] create-dmg not found; creating a standard DMG with hdiutil..."
  DMG_PATH="${DIST_DIR}/${APP_NAME}.dmg"
  STAGING_DIR="${DIST_DIR}/_dmg_staging"
  safe_rmtree "${STAGING_DIR}"
  if [[ -f "${DMG_PATH}" ]]; then
    rm -f "${DMG_PATH}"
  fi
  mkdir -p "${STAGING_DIR}"
  cp -R "${APP_PATH}" "${STAGING_DIR}/"
  ln -s /Applications "${STAGING_DIR}/Applications"
  hdiutil create \
    -volname "${APP_NAME}" \
    -srcfolder "${STAGING_DIR}" \
    -ov \
    -format UDZO \
    "${DMG_PATH}"
  safe_rmtree "${STAGING_DIR}"
  echo "Built installer: ${DMG_PATH}"
fi

if command -v shasum >/dev/null 2>&1; then
  DMG_DIR="$(dirname "${DMG_PATH}")"
  DMG_NAME="$(basename "${DMG_PATH}")"
  (
    cd "${DMG_DIR}"
    shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256"
  )
  echo "Built checksum: ${DMG_PATH}.sha256"
fi

echo "[5/5] Done."
