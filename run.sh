#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$root/.cli-conda" ]]; then
  mapfile -t cli < "$root/.cli-conda"
  if [[ ! -x "${cli[0]:-}" || ! -d "${cli[1]:-}/conda-meta" ]]; then
    printf 'The Conda env recorded in %s is unavailable. Use the installed checkout or rerun its installer.\n' "$root/.cli-conda" >&2
    exit 1
  fi
  export PATH="$(dirname -- "${cli[0]}"):$PATH"
  # Start Conda without the caller's activation stack or its deactivation hooks.
  unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV
  exec "${cli[0]}" run --no-capture-output -p "${cli[1]}" python "$@"
fi
python="$root/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  printf 'Multiview Data Python is missing in %s. Use the installed checkout or run %s/install.sh.\n' "$root" "$root" >&2
  exit 1
fi
exec "$python" "$@"
