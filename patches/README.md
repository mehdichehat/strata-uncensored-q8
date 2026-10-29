# The local changes to Strata, and how to keep them

This folder holds the changes this copy of Strata has on top of the upstream release it was downloaded from
([github.com/Niko1221/Strata](https://github.com/Niko1221/Strata), version 0.1.43, upstream commit
`7695859a` of 2026-10-11), and the two tools that keep them.

Two sets of changes are here, plus the tools that keep them:

| Patch | What it changes | Why |
| --- | --- | --- |
| `uncensored-q8-mtp.patch` | `tools/iq_pack.py`, `include/strata/artifact/gguf_reader.hpp`, `src/core/native_dense.cpp` (changed), `tools/mtp_from_gguf.py` (new), `docs/UNCENSORED_Q8.md` (new) | the MTP awareness that lets the Qwen3.8-Flash-Next Uncensored Q8_0 pack run (see `docs/UNCENSORED_Q8.md`) |
| `arena-prepop.patch` | `src/core/pinned.cu`, `include/strata/core/pinned.hpp`, `tests/core/arena_populate_test.cpp` (new), `CMakeLists.txt`, `docs/DETAILS.md`, `docs/TROUBLESHOOTING.md` | the startup fix: the engine fills its host arena itself, in parallel, before the CUDA driver pins it (measured 455 s -> 23.7 s for the fill on this PC) |
| `persistence.patch` | `setup.py`, `docs/UNCENSORED_Q8.md` (the section below), `tools/test_setup_local_changes.py` (new), `local/README.md`, `local/install-local-engine.py`, `local/build-local-engine.sh` | this folder: the engine stays the one built from these sources, and a new copy of Strata says which patches it has to apply before it compiles the upstream source |

All three patches apply to a fresh upstream copy of commit `7695859a` with `git apply`, in that order (the second
one on top of the first, the third on top of both). They were checked that way: a fresh checkout of `7695859a`
plus the three patches is byte-identical to this tree's changed files.

## The engine stays the one built from these sources

`setup.py` normally downloads a ready-made engine and replaces the installed one when it is older than the
published one. That would drop the changes above. Three files in `setup.py` make the engine in `engine/` the one
that is built here:

- `engine/BUILD.json` has `"source": "local"` - `get_prebuilt()` returns `None` for it (`setup.py:2973`), so
  nothing is downloaded;
- the same file's `"src"` is the hash of these sources (`source_hash(ENGINE_SOURCES)`), so
  `update_installed_engine()` (`setup.py:3127`) compiles the engine again only when `src/`, `include/` or
  `CMakeLists.txt` changed - and compiles it from here, not downloads it;
- `build_engine()` (`setup.py:3402`) then finds the engine up to date and starts it as it is.

`local/install-local-engine.py` writes that file, in setup.py's own format and with setup.py's own hash, so the
two always agree. It also copies the compiled engine to `engine/strata` and points the model's config at it.

**After every change in `src/` or `include/`:**

    local/build-local-engine.sh          # compile (incremental) and install

Then `./setup.sh` starts the new engine. Without the install, the next setup run would notice the changed source
and compile the engine itself (same result, but 10-20 minutes instead of a few).

## When a newer Strata comes out

This copy was not made with git, so `./update.sh` cannot fetch anything into it. To take the new upstream code
and keep these changes:

    git clone https://github.com/Niko1221/Strata.git new-strata
    cd new-strata
    git checkout 7695859a                  # the commit these patches were made against
    git apply /path/to/strata-local-backup/uncensored-q8-mtp.patch
    git apply /path/to/strata-local-backup/arena-prepop.patch
    git apply /path/to/strata-local-backup/persistence.patch

Then move the new upstream code in: the patches touch `src/`, `include/`, `tools/`, `tests/`, `setup.py` and
`docs/`, so `git merge` of the new main into that branch shows the conflicts where upstream changed the same
lines. The model files, the pack and the config are not in the repository and are not touched by this.

## Copies of these files outside the repository

A copy of the patches and of the changed files is in `/home/chota/lmstudio/strata-local-backup/` (outside this
folder, so it survives a new copy of Strata unzipped next to it):

- `uncensored-q8-mtp.patch`, `arena-prepop.patch`, `persistence.patch` - the three patches;
- `files/` - the changed files themselves, and `tree-changes-2026-10-21.tar.gz` - the same as one archive.
