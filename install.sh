#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

# Build from a fresh activation context; the caller's shell is unchanged.
unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV

case "${1:-}" in
  -h|--help) echo 'Usage: ./install.sh (set VCI_CLI_ENV to choose the Conda environment)'; exit 0 ;;
esac
[[ $# -eq 0 ]] || { echo 'Unexpected arguments; see ./install.sh --help.' >&2; exit 2; }

# Also works when distributed as a source archive with no declared submodules.
if [[ -e .git || -f .gitmodules ]]; then
  git submodule update --init --recursive
fi
if [[ -n "${VCI_CONDA_EXE:-}" ]]; then
  conda="$VCI_CONDA_EXE"
elif [[ -x "$root/../.miniforge/bin/conda" ]]; then
  conda="$root/../.miniforge/bin/conda"
else
  conda="$(command -v conda || true)"
fi
if [[ -z "$conda" ]] || ! command -v -- "$conda" >/dev/null 2>&1; then
  echo 'Conda is required. Put Miniforge or Miniconda on PATH, or set VCI_CONDA_EXE.' >&2
  exit 1
fi
conda="$("$conda" info --base)/bin/conda"
export PATH="$(dirname -- "$conda"):$PATH"
cli_env="${VCI_CLI_ENV:-vci-multiview-data}"
envs="$("$conda" env list)"
if ! awk -v name="$cli_env" '$1 == name {found=1} END {exit !found}' <<< "$envs"; then
  "$conda" create -y -n "$cli_env" "python=${PYTHON_VERSION:-3.10}" pip
fi
prefix="$("$conda" run -n "$cli_env" python -c 'import sys; print(sys.prefix)')"
"$conda" run --no-capture-output -p "$prefix" python -m pip install -e ".[native]"
"$conda" run --no-capture-output -p "$prefix" python -c 'import cv2, h5py, multiview_data'
printf '%s\n%s\n' "$conda" "$prefix" > .cli-conda
printf '\nInstalled CLI environment: %s\n' "$prefix"
printf '\nTry: %s\n' './run.sh examples/inspect_sequence.py --help'
