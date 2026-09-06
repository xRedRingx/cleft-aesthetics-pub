#!/usr/bin/env bash
#
# Run the test suite in the pinned environment. This is THE way to run the tests.
#
#   scripts/test.sh                     # whole suite
#   scripts/test.sh -k resume           # anything after the script name goes to pytest
#   scripts/test.sh tests/test_guards.py -x
#
# Creates .venv on the declared python series if it is absent or on the wrong
# series, installs the pinned extra when pyproject.toml has changed, and runs
# pytest inside it. Typing `pytest` directly instead uses whatever interpreter and
# whatever numpy happen to be on PATH; tests/test_environment.py is the backstop
# that catches that, but this script is what makes it not happen in the first
# place.
#
# The python series is read from docker/requirements.txt rather than hardcoded, so
# there is one source of truth for it.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

die() { echo "test.sh: FATAL: $*" >&2; exit 1; }

# --------------------------------------------------------------------------
# what interpreter does this project want?
# --------------------------------------------------------------------------

SERIES="$(sed -n 's/^#[[:space:]]*python:[[:space:]]*\([0-9][0-9]*\.[0-9][0-9]*\)\..*$/\1/p' \
  docker/requirements.txt | head -1)"
[[ -n "$SERIES" ]] || die "docker/requirements.txt has no '# python: X.Y.Z' line"

series_of() {
  "$1" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))' 2>/dev/null || true
}

find_python() {
  for candidate in "python${SERIES}" python3 python; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if [[ "$(series_of "$candidate")" == "$SERIES" ]]; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

# --------------------------------------------------------------------------
# venv
# --------------------------------------------------------------------------

VENV="${CLEFT_VENV:-$REPO/.venv}"
VPY="$VENV/bin/python"

if [[ -x "$VPY" && "$(series_of "$VPY")" != "$SERIES" ]]; then
  echo "test.sh: $VENV is python $(series_of "$VPY"), this project wants $SERIES - recreating"
  rm -rf "$VENV"
fi

if [[ ! -x "$VPY" ]]; then
  BASE="$(find_python)" || die "$(cat <<EOF
no python $SERIES on PATH.

Tried: python${SERIES}, python3, python.

  Debian/Ubuntu : sudo apt install python${SERIES} python${SERIES}-venv
  pyenv         : pyenv install ${SERIES}.13 && pyenv local ${SERIES}.13

Do NOT substitute another series. numpy 1.26.4 publishes no wheels beyond 3.12,
and the cluster image ships $SERIES.
EOF
)"
  echo "test.sh: creating $VENV from $BASE (python $SERIES)"
  "$BASE" -m venv "$VENV"
  "$VPY" -m pip install --upgrade pip --quiet
fi

# --------------------------------------------------------------------------
# dependencies, reinstalled only when the pins change
# --------------------------------------------------------------------------

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else "$VPY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"
  fi
}

STAMP="$VENV/.cleft-pins"
WANT="$(hash_file pyproject.toml)"

if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$WANT" ]]; then
  echo "test.sh: installing pinned dependencies"
  "$VPY" -m pip install -e ".[test]" --quiet
  echo "$WANT" > "$STAMP"
fi

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

echo "test.sh: $("$VPY" -VV | head -1)"
exec "$VPY" -m pytest "$@"
