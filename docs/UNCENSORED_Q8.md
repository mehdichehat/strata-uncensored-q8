# Uncensored Q8_0-MTP (5 shards, integrated draft block)

**Status: working.** A community Q8_0 quantization with an integrated MTP draft block
(`qwen4exp.block_count = 49`, `qwen4exp.nextn_predict_layers = 1`) runs on the
native GGUF path after three small MTP-awareness fixes (packer, arch guard, dense
walk). Validated on 4x RTX 3090 (sm_86) + 256 GB DDR4, Linux, CUDA 12.4, engine
0.1.43 + local patches.

> **On this page:** [What is in the file](#what-is-in-the-file) · [The pack](#the-pack-by-hand) ·
> [The server](#the-server) · [Speed](#what-to-expect) · [Quality](#quality) ·
> [What the engine did not have](#what-the-engine-did-not-have-and-has-now) ·
> [Keeping these changes](#keeping-these-changes) · [Known issues](#known-issues)

## What is in the file

Five shards, ~179 GB together (`Qwen3.8-Flash-Next-Uncensored-Q8_0-MTP-0000N-of-00005.gguf`),
plus `mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf` (866 MB vision encoder).

`tools/quantscope.py` on the folder (header ground truth, not the file names):

| Group | Weights | GB | Types |
| --- | --- | --- | --- |
| Routed experts (49 blocks) | 123.31B | 131.02 | Q8_0 100% |
| PLE table (`per_layer_token_embd.weight` [160, 320001536], shard 2 alone) | 51.23B | 54.44 | Q8_0 100% |
| Attention, hyper-connections, token embedding, output head, SSM | — | ~4.3 | Q8_0 ~100% |
| Routers, norms | — | ~0.3 | F32 |
| MTP draft (blk.48: full MoE + `nextn.*`, 29 tensors) | 0.01B | ~0.0 | Q8_0 / F32 |

- Every expert role in every block is Q8_0 (ggml type 8): gate/up/down `8/8` in all
  49 blocks. No mixed types, no per-role shard splits (`native_experts.txt` v3).
- Block 48 is the integrated MTP draft: `blk.48.ffn_*_exps` (same [2560, 640, 512]
  shapes as the main blocks) plus `blk.48.nextn.{hnorm,enorm,eh_proj,hc_head_*}`.
  The target forward runs blocks 0-47 (48 layers); the draft is served from `mtp/rt`.
- `qwen4exp.expert_count = 512`, `expert_used_count = 10`, `embedding_length = 2560`.

## The pack (by hand)

```sh
.venv/bin/python tools/iq_pack.py \
  --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-Q8_0-MTP-00001-of-00005.gguf \
  --out packs/uncensored-q8 --compat-bf16
```

Under a minute. Writes `index.txt` (1079 tensors, 303 served natively, arena
1.38 GiB), `dense.bin` (1.3 GB), `native_experts.txt` (48 rows, all `8 8`,
131 GB total), the tokenizer, `conversions.json` (483 converted: 96 exact,
387 rounded to BF16, max |err| 0.0144) and `compat-bf16.json`.
**`--compat-bf16` is required** (F32 routers/norms, Q8_0 hyper-connection
projections the engine reads as BF16). Do not add `--experts-bin`: the 128 GB
of experts stay in the GGUF files. The packer prints two `mtp:` lines: the
excluded draft block and the 29 draft tensors left in the GGUF.

## The server

`--mtp` is the **integrated** draft (`mtp/rt-uncensored`, built from blk.48 by
`tools/mtp_from_gguf.py` — no download). The base draft (`mtp/rt`, section below)
stays on disk as a fallback; one `--mtp` path switch compares them.

## Integrated draft (`tools/mtp_from_gguf.py`)

`--mtp mtp/rt-uncensored`, built from the file's own blk.48:

```sh
.venv/bin/python tools/mtp_from_gguf.py \
  --gguf /path/to/...-00001-of-00005.gguf --out mtp/rt-uncensored
cp data/draft_vocab.bin mtp/rt-uncensored/draft_vocab.bin
```

The runtime contract (`dense.txt` names/kinds/dims) is set-for-set identical to
`mtp_rt.py`'s 29 lines. Experts fuse gate/up along rows (lossless) and requant to
the draft Q2_0 grid (draft-only rounding); Q8_0 dense is reused byte for byte;
`index_qk_proj` concats q/k rows and `fc_embedding`/`fc_hidden` split `eh_proj`
along columns (both lossless). Norms take mtp_rt's Gemma +1; hc/mixer/router go
Q8_0/F32 → BF16 (ggml nearest-even).

- blk.48 is a genuinely different draft from the base (rel RMS 0.43 on gate
  expert 0 — not quant noise), so this is not a re-encoding of the base MTP.
- eh order falsified live: `--swap-eh` (hid,emb) drafts **0/0 tokens**; the
  default emb,hid order drafts at 66%. A wrong mapping fails loudly in
  acceptance, never silently in quality (the target verifies every token).
- Acceptance: **66%** (93/140) vs the base draft's 79% — the cost of Q8_0→Q2_0
  double quantization plus fine-tune drift. ~10% decode slower than base-draft
  config; everything else identical. Switch back with one `--mtp` path change.

## Base draft (fallback, `mtp/rt`)

Once (base MTP via range requests + local pack, ~0.9 GB):

```sh
.venv/bin/python tools/mtp_fetch.py inventory --out mtp-fetch
.venv/bin/python tools/mtp_fetch.py fetch --out mtp-fetch
.venv/bin/python tools/mtp_fetch.py verify --out mtp-fetch
mkdir -p mtp
.venv/bin/python tools/mtp_pack.py --src mtp-fetch --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

79% acceptance on this target (166/210). One `--mtp` switch to compare.

`strata-uncensored-q8.json` (4x 24 GB, 256 GB RAM — everything fits, no RAM budget):

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/uncensored-q8",
    "--native", "/path/to/...-00001-of-00005.gguf",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt-uncensored",
    "--pipeline-windows", "2",
    "--kv", "k8v4",
    "--max-context", "32768"
  ],
  "gpu": [0, 1, 2, 3],
  "layer_split": "auto",
  "tokenizer": "packs/uncensored-q8/tokenizer",
  "model_name": "qwen3.8-flash-next-uncensored-q8_0",
  "vision": {
    "exe": "engine/strata-vision",
    "mmproj": "/path/to/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf",
    "model": "/path/to/...-00001-of-00005.gguf",
    "gpu": false, "threads": 32, "max_tokens": 1024
  }
}
```

The vision encoder (`tools/vision`, `STRATA_VISION_CUDA=ON`, sm_86) is a separate
build; images verified end-to-end with the uncensored mmproj (CPU encoder first;
GPU needs VRAM headroom the 4-way split does not currently leave).

## What to expect

4x RTX 3090, EPYC 7532 (AVX2, no AVX-512), 256 GB DDR4, layer split `K=10,23,35`
without trim (`K=11,23,35`), caches ~14.4k/24.6k pairs (~93% routed mass):

| Step | Short-chat decode | Notes |
| --- | --- | --- |
| 1 GPU, no MTP | 8.7 tok/s | gate test only (1500 cache slots) |
| 4-GPU split, no MTP | 59-70 tok/s | suffix drafts only, hit rate 96-99% |
| + base MTP (`mtp/rt`) | 79 tok/s | drafts 166/210 (79%) |
| + `--pipeline-windows 2` | 85 tok/s | 4 stages overlap; text-identical to serial |
| + `--kv k8v4`→`int8` + resident 32K | 89 tok/s | 262K needs streaming; int8 is the lossless default |
| + calibrate (pcie 0, 8 workers, adapt 1/160/0.97, spec-min-p 0.7) | ~105 tok/s | `tools/calibrate.py` verdict below |
| + integrated MTP (`mtp/rt-uncensored`, current) | ~78 tok/s fresh | drafts 93/140 (66%); file's own draft |

Prompts: 4K ~870 tok/s, 16K ~1970 tok/s, 32K ~2480 tok/s, 253K ~3180 tok/s.
Short chat decode 72.5 tok/s bench median (integrated draft; base-draft config
measured 85.5). Reports: `bench/results/2026-10-13-community-4x-rtx-3090-*`.

## 262K context + int8 KV (the end state)

`--max-context 262144 --kv int8 --kv-resident 32768`: KV streaming keeps 32K
cells/layer in VRAM, the rest in 0.52 GiB of pinned RAM. Native 262144 needs no
RoPE flag. Measured:

| Test | Result |
| --- | --- |
| 253,256-token prompt, fresh | 3234 tok/s prefill (78 s), then 80 tok/s decode |
| Needle (`BLUE-DRAGON-42` at 50% of 253K) | retrieved exactly |
| Same prompt again (prompt cache 253,251 reused) | 0.9 s total, 107 tok/s decode |

## Calibrate verdict (this box: EPYC 7532 + 4x3090, Q8)

- `--pcie-frac 0` (+4% over default 0.55: hit rates are 96-99%, PCIe share hurts)
- `--pool-workers 8` (31: 98.0, 21: 100.3, 16: 103.4, **8: 104.6** — bandwidth-bound)
- `--adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` (101.9 vs 98.1 default)
- `--spec-min-p 0.7` (kept)

## Quality

- `native_expert_parity --synthetic q8_0/q8_0`: CPU rel 1.27e-2 and GPU rel
  1.27e-2 vs float (quantization noise, same band as Q4_0's 1.1-1.3e-2),
  CPU-vs-GPU 1.09e-7, GPU dequant vs ggml `to_float` bit-exact (0 of 4,915,200
  differ). **0 failures.**
- MTP acceptance 79% (166/210) with the base draft on this uncensored target —
  the target distribution aligns; output tokens are always target-verified.
- Generations checked by hand (LRU-cache coding prompt: correct; image
  description: correct). Teacher-forced KL vs llama.cpp on this file: not yet
  measured.

## What the engine did not have, and has now

Files with an integrated MTP draft block (`qwen4exp.nextn_predict_layers = 1`,
`block_count = 49`):

- **Packer** (`tools/iq_pack.py`): `expert_layout` packs only the main blocks
  (`n_layers = max_blk + 1 - nextn`); `index_standalone` leaves the draft
  block's 29 tensors in the GGUF for the mtp tools instead of indexing
  unloadable shape-only rows. Files without the key pack byte-for-byte as before.
- **Arch guard** (`include/strata/artifact/gguf_reader.hpp`): `check_architecture`
  expects `block_count = 48 + nextn_predict_layers` (absent key: unchanged).
- **Dense walk** (`src/core/native_dense.cpp`): `load`, `weight_bytes_for` and
  `served_names` skip the draft blocks (`draft_cut`/`is_draft` from the shards'
  own metadata), keeping the three walks in step. Geometry stays 48 everywhere.

## Keeping these changes

The three fixes above (and `tools/mtp_from_gguf.py`) are in this tree, not in
upstream Strata. They are kept as patches in `local/` - `local/README.md` has the
whole story:

| Patch | Change |
| --- | --- |
| `local/uncensored-q8-mtp.patch` | the packer, the arch guard, the dense walk, `tools/mtp_from_gguf.py`, this file |
| `local/arena-prepop.patch` | the startup fix (the engine fills its host arena itself before the CUDA driver pins it) |
| `local/persistence.patch` | the setup's warning for a new copy of Strata, and the files in `local/` |

The engine that runs is the one built from this tree: `engine/BUILD.json` says
`"source": "local"`, so `setup.py` never downloads the published engine over it,
and compiles it again only when `src/` or `include/` changed
(`local/install-local-engine.py` writes that file, `local/build-local-engine.sh`
compiles and installs it).

A new copy of Strata - a new release unzipped elsewhere - sets itself up like
this install but compiles the upstream source, which has none of these fixes: it
says so and names the patches to apply to it first. Copies of the patches are in
`/home/chota/lmstudio/strata-local-backup/`.

## Known issues

- `STRATA_STAGE_TRIM=1` on this 4-way native-Q8 split: the engine dies silently
  during cache fill (log ends after the `GPU 0` line, no error; without trim the
  same config loads). Trim predicted 96.1% cached vs 92.9% and -16%/window, so
  it is worth retrying on a newer engine with a debugger attached. Without trim
  everything above holds.
- The integrated blk.48 draft is live (`mtp/rt-uncensored`, 66% acceptance vs the
  base's 79% — Q8_0→Q2_0 double quantization plus fine-tune drift). The base
  draft stays as a one-switch fallback. Revisit if acceptance drops on some
  workload.
- `--pipeline-windows 2` and `--adapt-async 1` are mutually exclusive on 3+
  stages (the engine turns the pipeline off with a one-line note). Measured
  here: pipeline 85.2 vs async 80.6 tok/s — pipeline kept.
- CPU has no AVX-512 (EPYC Zen 2): expert miss path runs AVX2 kernels.
  `tools/calibrate.py` verdict: `--pcie-frac 0 --pool-workers 8 --spec-min-p 0.7
  --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97` (all in the live config).

## Speed vs quality: what costs what

Short answer: the speed comes from doing less work per token, not from cheaper
weights — output quality is the target's own everywhere it matters:

- **MTP speculative decoding** (either draft): quality-neutral by construction.
  The target verifies every draft token; accepted or not, final tokens equal a
  non-speculative run. Acceptance (66% integrated, 79% base) sets speed only.
- **`--pipeline-windows 2`**: no quality trade by design — "the tokens are the
  serial loop's" (docs/MULTI_GPU.md), verified identical greedy text.
- **`--kv int8` + streaming**: int8 KV is the default fidelity (the lossy
  opt-ins are q4/k8v4's 4-bit V). 253K needle retrieved exactly; a repeated
  253K prompt reuses 253,251 cached tokens with identical output.
- **`--pcie-frac 0`**: misses run on the CPU's ggml-exact kernels instead of GPU
  approximations — if anything more faithful (measured CPU-GPU gap: 1e-7).
- **Workers/adapt/prefill-help**: same kernels, different scheduling/placement.
  GPU-vs-CPU placement can flip near-ties (last-bit rounding), same class as
  the default adaptive tier — upstream "same quality" (+ cache-parity bench).
- **Defaults left on**: route-tail-skip (skips rank≥7 routed misses, +10-20%
  decode; the one documented token-changer) and prefill CPU share (mean KL
  ~0.004). Disable with `STRATA_ROUTE_TAIL_SKIP=0` / `STRATA_PREFILL_CPU_SHARE=0`
  for a paranoia A/B — if outputs match, nothing was lost.
- **`--compat-bf16`**: one-time max err 0.0144 on small projections only (same
  profile as UD-Q4_K_XL); experts and PLE table are byte-identical to the GGUF.

## Strata vs llama.cpp on the same file (2026-10-13)

llama-server 0.3.0-dev (build 10802) with the owner's daily-driver flags
(`--spec-type draft-mtp --spec-draft-n-max 2 --split-mode layer --fit on
--n-cpu-moe 45 -t 32 --ctx-size 262144 --cache-type-k/v q8_0 --rope-scaling yarn
--rope-scale 32`), same prompts, greedy both sides. Raw captures:
`bench/results/2026-10-13-community-4x-rtx-3090-uncensored-q8-0-integrated-262k/cmp_*.json`
(speed set) and `grade_*.json` (executed-code set).

| Task | llama.cpp | Strata (this config) |
| --- | --- | --- |
| Needle 8K (`BLUE-DRAGON-42`) | found | found |
| Needle 253K | not run (108 tok/s prefill → ~40 min) | **exact** |
| Math (75; 541) | 75 ✓ / 541 ✓ | 75 ✓ / 541 ✓ |
| Factual (Ouagadougou) | ✓ | ✓ |
| LRU, executed | PASS | PASS |
| fib, executed | PASS (byte-identical code both sides) | PASS |
| Decode | 24-29 tok/s | **65-109 tok/s (2.5-3.5x)** |
| Prefill | 25-124 tok/s | **110-3180 tok/s (9-30x)** |
| Draft acceptance | 86% (n-max 2) | 66-71% (spec 4) |

No quality loss on Strata: every graded answer matches. The speed gap is
offloading, not weights — llama's fit keeps most of the 179 GB on CPU
(VRAM 6/5/5/16 GB) with ubatch 512 on a bandwidth-bound EPYC, while the split
holds ~93% of routed experts on the GPUs. (Llama was not tested at 253K input;
its yarn-32 RoPE vs Strata's native 262144 is a real long-context difference —
both retrieve exactly at the ranges measured.)
