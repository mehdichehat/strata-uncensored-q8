# Measured speeds

Source: the live server log of this exact config (`strata-uncensored-q8.log`),
**32,727 completions** served over days of real agentic use (OpenCode coding
sessions with ~150 k-token prompts, not synthetic benchmarks).

Parse the engine's own `strata serve:` lines the same way to reproduce:

```
strata serve: prompt 162815 tokens = 161359 reused + 1456 read in 3303 ms (440.8 tok/s), 200 generated in 2172 ms (92.1 tok/s), drafts accepted 111 of 140, 6 checkpoints
```

## Completion speed (generation)

32,242 completions of ≥ 50 tokens:

| Stat | tok/s |
| --- | --- |
| median | **98.4** |
| mean | 100.3 |
| p90 | 120.2 |
| max | 180.5 |
| min | 33.2 |

The distribution is tight because the MTP draft carries it:

## Speculative decoding (MTP)

- Draft acceptance: **20,616,673 / 25,341,070 = 81.4 %** across the whole log.
- `--spec 4 --spec-min-p 0.7`: four draft tokens proposed per step, accepted
  greedily above p=0.7.

## Prefill

- Warm prompts are KV-cache hits (the `reused` column — typically >99 % of a
  150 k-token prompt re-read in a few hundred ms).
- Cold prefill (requests with > 500 genuinely new tokens, n = 12,316):
  median **436 tok/s**, max 3,786 tok/s.

## Hardware / software

| | |
| --- | --- |
| GPUs | 4x RTX 3090 (24 GB, sm_86) |
| Host | 256 GB DDR4, 32 threads used by the CPU vision path |
| OS / CUDA | Linux, CUDA 12.4 |
| Engine | Strata 0.1.43 (upstream `7695859a`) + the three patches in `patches/` |
| Model | Qwen3.8-Flash-Next Uncensored Q8_0-MTP, 5 shards ≈ 179 GB |
| Context | 262,144 tokens, KV int8, 32 k resident |

## Startup

The `arena-prepop.patch` is what makes this usable at all on this box: the
engine pre-populates its host arena in parallel before the CUDA driver pins
it — measured **455 s → 23.7 s** for the fill (see `patches/README.md`).
