#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
NNUNET_RAW="${NNUNET_RAW:-${nnUNet_raw:-}}"
NNUNET_PREPROCESSED="${NNUNET_PREPROCESSED:-${nnUNet_preprocessed:-}}"
NNUNET_RESULTS="${NNUNET_RESULTS:-${nnUNet_results:-}}"
DATASET=""

pick_first_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

resolve_conda_root() {
  if [[ -n "${CONDA_ROOT}" ]]; then
    printf '%s\n' "${CONDA_ROOT}"
    return 0
  fi

  if [[ -n "${CONDA_EXE:-}" ]]; then
    printf '%s\n' "$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
    return 0
  fi

  pick_first_existing_dir \
    "${HOME}/miniconda3" \
    "${HOME}/anaconda3"
}

resolve_conda_env_name() {
  if [[ -n "${CONDA_ENV_NAME}" ]]; then
    printf '%s\n' "${CONDA_ENV_NAME}"
    return 0
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    printf '%s\n' "${CONDA_DEFAULT_ENV}"
    return 0
  fi

  printf '%s\n' "nnunetv2"
}

resolve_nnunet_path() {
  local current="$1"
  shift
  if [[ -n "${current}" ]]; then
    printf '%s\n' "${current}"
    return 0
  fi
  pick_first_existing_dir "$@"
}

usage() {
  cat <<'EOF'
Usage:
  nnunetv2-doctor.sh [--dataset 22|Dataset022_FLARE22]

Environment overrides:
  CONDA_ROOT
  CONDA_ENV_NAME
  NNUNET_RAW
  NNUNET_PREPROCESSED
  NNUNET_RESULTS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|-d)
      DATASET="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

CONDA_ROOT="$(resolve_conda_root || true)"
CONDA_ENV_NAME="$(resolve_conda_env_name)"
NNUNET_RAW="$(resolve_nnunet_path "${NNUNET_RAW}" "${HOME}/nnUNet_raw" || true)"
NNUNET_PREPROCESSED="$(resolve_nnunet_path "${NNUNET_PREPROCESSED}" "${HOME}/nnUNet_preprocessed" || true)"
NNUNET_RESULTS="$(resolve_nnunet_path "${NNUNET_RESULTS}" "${HOME}/nnUNet_results" || true)"

if [[ -z "${CONDA_ROOT}" || ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda activation script: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

if [[ -n "${NNUNET_RAW}" ]]; then
  export nnUNet_raw="${NNUNET_RAW}"
fi
if [[ -n "${NNUNET_PREPROCESSED}" ]]; then
  export nnUNet_preprocessed="${NNUNET_PREPROCESSED}"
fi
if [[ -n "${NNUNET_RESULTS}" ]]; then
  export nnUNet_results="${NNUNET_RESULTS}"
fi

echo "=== Conda ==="
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "python=$(command -v python)"
echo "nnUNetv2_predict=$(command -v nnUNetv2_predict || true)"
echo

echo "=== nnUNet Paths ==="
echo "nnUNet_raw=${nnUNet_raw:-}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed:-}"
echo "nnUNet_results=${nnUNet_results:-}"
echo

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi
echo

echo "=== Torch CUDA ==="
python - <<'PY'
import os
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
try:
    print(f"device_count={torch.cuda.device_count()}")
except Exception as exc:
    print(f"device_count_error={exc!r}")
try:
    if torch.cuda.device_count() > 0:
        print(f"device0_name={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"device0_name_error={exc!r}")
for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    print(f"{key}={os.environ.get(key)}")
PY
echo

if [[ -n "${DATASET}" ]]; then
  echo "=== Dataset Lookup ==="
  python - "${DATASET}" <<'PY'
import os
import sys
from pathlib import Path

from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

dataset_arg = sys.argv[1]
dataset_name = maybe_convert_to_dataset_name(dataset_arg)
print(f"dataset={dataset_name}")

roots = {
    "raw": Path(os.environ["nnUNet_raw"]),
    "preprocessed": Path(os.environ["nnUNet_preprocessed"]),
    "results": Path(os.environ["nnUNet_results"]),
}
for label, root in roots.items():
    path = root / dataset_name
    print(f"{label}_path={path} exists={path.exists()}")

results_dir = roots["results"] / dataset_name
if results_dir.exists():
    print("model_folders:")
    for child in sorted(results_dir.iterdir()):
        if child.is_dir():
            print(f"  {child.name}")
PY
fi
