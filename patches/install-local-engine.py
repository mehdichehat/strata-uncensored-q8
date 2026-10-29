#!/usr/bin/env python3
"""Install the engine compiled from this source tree as this PC's engine (engine/strata + engine/BUILD.json).

Why this file exists
--------------------
setup.py downloads a ready-made engine (the published one) whenever engine/ has no BUILD.json, and replaces the
installed one when it is older than the published one.  A download would replace the engine built from THESE
sources with the published binary - the local MTP / uncensored-Q8 patches (docs/UNCENSORED_Q8.md) would be gone.

engine/BUILD.json with "source": "local" is how setup.py knows the engine in engine/ was compiled here:
  - get_prebuilt() returns None for it (setup.py:2973) - nothing is downloaded, ever;
  - update_installed_engine() (setup.py:3127) compiles it again only when the source of these files changed;
  - build_engine() (setup.py:3402) then says "engine already built for this PC" and starts it as it is.
So the engine that runs is always the one built from this tree, and a change in src/ or include/ is picked up by
the next setup run (which compiles it again).

What it does
------------
  1. copies the compiled engine (build/strata) to engine/strata;
  2. writes engine/BUILD.json in setup.py's own format, with setup.py's own source hash (so the hashes always
     agree with what setup.py compares against);
  3. points the model's config at that engine (absolute paths), so it runs the same engine from any folder.

Use it after every compile:
    .venv/bin/python local/install-local-engine.py
or compile and install in one step:
    local/build-local-engine.sh
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402  (the canonical owner of the engine bookkeeping)

BUILT = ROOT / "build" / setup.EXE
ENG = setup.engine_dir(13)
CONFIGS = sorted(ROOT.glob("strata-*.json"))


def cuda_lib_dirs() -> list[str]:
    """Where the CUDA runtime the engine loads lives (the 'cuda_dirs' of setup.py's BUILD.json)."""
    out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    dirs: list[str] = []
    for line in out.splitlines():
        if "=>" not in line:
            continue
        if "libcudart.so" not in line and "libcublas.so" not in line:
            continue
        path = line.split("=>")[-1].strip().split(" (")[0]
        folder = str(Path(path).parent)
        if folder not in dirs:
            dirs.append(folder)
    return dirs or setup.cuda_lib_dirs(13)


def archs_of(cfg: dict) -> list[int]:
    """The compute capabilities the engine needs code for: the cards the model runs on."""
    archs: set[int] = set()
    cards = cfg.get("gpu")
    cards = cards if isinstance(cards, list) else [cards]
    for index in cards:
        if index is None:
            continue
        info = setup.gpu_info(int(index))
        if info:
            archs.add(int(info["arch"]))
    return sorted(archs)


def install(cfg_paths: list[Path]) -> dict:
    if not BUILT.exists():
        sys.exit(f"no compiled engine at {BUILT}: compile it first (local/build-local-engine.sh)")
    ENG.mkdir(exist_ok=True)
    # the copy is written next to it and moved over the old one: a running engine is never half replaced
    tmp = ENG / (setup.EXE + ".new")
    shutil.copy2(BUILT, tmp)
    os.replace(tmp, ENG / setup.EXE)

    meta = {
        "source": "local",
        "version": setup.source_version(),
        "archs": [],
        "vision": "none",
        "cuda_dirs": cuda_lib_dirs(),
        "src": setup.source_hash(setup.ENGINE_SOURCES),
        "vision_src": setup.source_hash(setup.VISION_SOURCES),
    }
    floor = setup.cpu_floor(setup.cpu_info()[1])
    if floor:
        meta["isa_floor"] = floor

    all_archs: set[int] = set()
    for path in cfg_paths:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        all_archs |= set(archs_of(cfg))
        if cfg.get("vision"):
            meta["vision"] = "gpu" if (ENG / setup.VEXE).exists() else "cpu"
    if all_archs:
        meta["archs"] = sorted(all_archs)
    (ENG / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def point_configs_at(cfg_paths: list[Path]) -> None:
    """The config runs the installed engine, from any folder (absolute paths)."""
    for path in cfg_paths:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        cfg["exe"] = str(ENG / setup.EXE)
        cfg["cwd"] = str(ROOT)
        if cfg.get("vision") and (ENG / setup.VEXE).exists():
            cfg["vision"]["exe"] = str(ENG / setup.VEXE)
        setup.write_config(path, cfg)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    cfg_paths = [Path(a) for a in args] or CONFIGS
    cfg_paths = [p for p in cfg_paths if p.exists()]
    if not cfg_paths:
        print("no model config found (strata-*.json); the engine is installed anyway")
    meta = install(cfg_paths)
    point_configs_at(cfg_paths)
    print(f"engine installed : {ENG / setup.EXE} (from {BUILT})")
    print(f"BUILD.json       : {json.dumps(meta)}")
    for path in cfg_paths:
        print(f"config           : {path.name} -> {json.loads(path.read_text(encoding='utf-8-sig'))['exe']}")
    # the checks setup.py itself makes, on the files just written
    same = setup.source_hash(setup.ENGINE_SOURCES) == meta["src"]
    print(f"setup.py sees    : source {'matches' if same else 'DIFFERS FROM'} the installed engine")
    print(f"                   (a change in src/ or include/ makes the next setup run compile it again)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
