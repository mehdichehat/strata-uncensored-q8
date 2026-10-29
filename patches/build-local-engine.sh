#!/bin/sh
# Compile the engine from this source tree and install it as this PC's engine (engine/strata + engine/BUILD.json).
#
# After a change in src/ or include/, this is what makes the next ./setup.sh run use it: the engine in engine/ is
# the one the model config points at, and its BUILD.json (written by local/install-local-engine.py) says it was
# compiled here - so setup.py never downloads the published engine over it.
#
# The compile is incremental (cmake): only the changed files are rebuilt.
set -e
cd "$(dirname "$0")/.."
BUILD_DIR="${BUILD_DIR:-build}"
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "no build folder in $BUILD_DIR: configure it first (see docs/DETAILS.md for the cmake line)." >&2
  exit 1
fi
echo "Compiling the engine in $BUILD_DIR ..."
cmake --build "$BUILD_DIR" --target strata -j "$(nproc)"
.venv/bin/python local/install-local-engine.py "$@"
