# Strata — Uncensored Q8_0 + MTP production setup

My production setup for running **Qwen3.8-Flash-Next Uncensored Q8_0-MTP**
(a ~179 GB, 5-shard community MoE quantization with an integrated MTP draft
block) on the [Strata](https://github.com/Niko1221/Strata) inference engine,
on a 4x RTX 3090 / 256 GB DDR4 box. This model serves my daily agentic coding
(OpenCode) at **~100 tok/s median with 81 % speculative-draft acceptance at a
150 k-token context** — measured numbers in [`benchmarks/speeds.md`](benchmarks/speeds.md).

Upstream Strata is MIT (C++/CUDA); this repo is not a fork dump — it holds
exactly the parts that make it run this model in production: the server
config, the engine patches, the packing tools, and the measured results.

## What is in here

| Path | What it is |
| --- | --- |
| `config/strata-uncensored-q8.json` | The live OpenAI-compatible server config (4 GPUs, MTP, int8 KV, 262 k context). Paths are host-specific. |
| `patches/` | Three `git apply`-able patches against upstream commit `7695859a` (v0.1.43) + the build/install tooling that keeps the locally built engine from being silently replaced by a download. [`patches/README.md`](patches/README.md) explains each one. |
| `tools/` | The IQ quant packer (`iq_pack.py`) and the MTP draft extractor (`mtp_from_gguf.py`) with their helper modules — the tools that produced the pack this config runs. |
| `docs/UNCENSORED_Q8.md` | The full write-up: what the file contains (tensor-level), the three engine fixes, expected speed, quality notes, known issues. |
| `benchmarks/speeds.md` | Speeds from 32,727 real completions in the server log, and how to reproduce the stats. |

## The three engine patches

1. **`uncensored-q8-mtp.patch`** — MTP awareness in three places (packer,
   arch guard, dense walk) so a Q8_0 pack with an integrated draft block
   (`qwen4exp.block_count=49`, `nextn_predict_layers=1`) loads and runs on the
   native GGUF path. Without it the engine rejects the file.
2. **`arena-prepop.patch`** — startup fix: the engine fills its pinned host
   arena itself, in parallel, before the CUDA driver pins it.
   **455 s → 23.7 s** boot time for the 179 GB model on this machine.
3. **`persistence.patch`** — makes `setup.py` treat the locally built engine
   as the source of truth (`engine/BUILD.json` `"source": "local"` + source
   hash), so an update never downloads a stock engine over the patched one.

All three apply cleanly, in order, to a fresh checkout of `7695859a` and were
verified byte-identical against the running tree.

## Reproducing on your own box

```bash
git clone https://github.com/Niko1221/Strata.git && cd Strata
git checkout 7695859a
git apply /path/to/patches/uncensored-q8-mtp.patch
git apply /path/to/patches/arena-prepop.patch
git apply /path/to/patches/persistence.patch
# build (CUDA 12.4, sm_86) — see patches/build-local-engine.sh
./setup.sh   # with a config/strata-uncensored-q8.json adapted to your paths
```

You will need the model shards and an expert profile (`data/expert-profile.bin`);
`tools/iq_pack.py` + `tools/mtp_from_gguf.py` show how the pack is built from
BF16 sources. The config's `exe`, `--native` and `--mtp` paths are absolute to
my machine — adapt them.

## Key config flags, and why

| Flag | Why |
| --- | --- |
| `--spec 4 --spec-min-p 0.7` | MTP speculative decoding; 81 % acceptance measured |
| `--mtp mtp/rt` | runtime draft blob extracted by `tools/mtp_from_gguf.py` |
| `--kv int8 --kv-resident 32768` | int8 KV cache, 32 k tokens resident — fits the 262 k context across 4 GPUs |
| `--max-context 262144` | agentic sessions run at 150 k+ tokens |
| `--expert-cache auto --pcie-frac 0` | experts stay on the GPUs; the PLE table streams from the pre-populated arena |
| `--adapt-*` | online expert-reordering adaptation (re-runs every turn, decay 0.97) |

## Results at a glance

- **98.4 tok/s median** completion (32,242 samples ≥ 50 tok; p90 120, max 180)
- **81.4 %** MTP draft acceptance (20.6 M / 25.3 M drafts)
- **436 tok/s median** cold prefill (12,316 samples > 500 new tokens)
- KV reuse makes warm 150 k-token prompts re-read in ~0.3–1.5 s
- Boot with 179 GB model: **23.7 s** (was 455 s before the arena fix)

## License

MIT — upstream copyright Niko1221 and the Strata contributors; my additions
under the same terms (see `LICENSE`).
