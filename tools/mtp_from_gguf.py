"""tools/mtp_from_gguf.py - the MTP draft runtime from a GGUF with an INTEGRATED draft block.

    python tools/mtp_from_gguf.py \
      --gguf /path/to/Qwen3.8-Flash-Next-Uncensored-Q8_0-MTP-00001-of-00005.gguf \
      --out mtp/rt-uncensored [--swap-eh]

Some files (qwen4exp.nextn_predict_layers = 1) carry their MTP draft as the last block
(blk.48 here) instead of needing tools/mtp_fetch.py. This builds the same mtp/rt layout
tools/mtp_rt.py writes (experts.bin in the engine blob layout, dense.bin + dense.txt),
from blk.N's GGUF-native tensors:

  experts      gate/up fused along rows (lossless byte concat of Q8_0 rows), requantized to the
               draft Q2_0 grid with mtp_pack's MSE scale search; down likewise. Draft-only rounding.
  dense Q8_0   attention, shared expert, indexer projections: reused byte for byte. The fused
               indexer.index_qk_proj is q_proj rows ++ k_proj rows (512 + 128 = 640: lossless concat).
               fc_embedding/fc_hidden split nextn.eh_proj along columns (blocks 0:80 / 80:160 of each
               row: lossless); --swap-eh takes hid,emb order when the default emb,hid order drafts nothing.
  dense BF16   hyper-connection + mixer projections (Q8_0 dequant) and the F32 router, ggml's
               round-to-nearest-even (iq_pack.bf16_bytes), like --compat-bf16.
  dense F32    norms as F32 with the Gemma +1 (mtp_rt's rule).

The concatenation/split orders are falsifiable: a wrong one drafts ~0% and the target still verifies
every token, so output quality never depends on them - only speed does. Acceptance >= the base draft's
(~79%) confirms the mapping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
from gguf import GGMLQuantizationType as Q, quants  # noqa: E402

from iq_pack import Model, bf16_bytes  # noqa: E402 (Model: split shards; bf16_bytes: ggml BF16 rounding)
from mtp_pack import q2_0  # noqa: E402 (draft Q2_0 grid, MSE scale search)
from mtp_rt import blob_of, H, FF, NE  # noqa: E402 (engine blob layout; H=2560 FF=640 NE=512)

BLK = "blk.48."
Q8_ROW = 34  # bytes per 32 values


def q8_rows(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Raw Q8_0 bytes [rows, cols//32 * 34] -> float32 [rows, cols] (ggml reference dequant)."""
    assert raw.size == rows * (cols // 32) * Q8_ROW, (raw.size, rows, cols)
    return quants.dequantize(raw, Q.Q8_0).astype(np.float32).reshape(rows, cols)


def expert_bytes(model: Model, name: str, e: int, rows: int, cols: int) -> np.ndarray:
    """Expert e's raw Q8_0 bytes of a [cols, rows, NE] GGUF tensor."""
    g, t, mm, _ = model.where[name]
    row_bytes = cols // 32 * Q8_ROW
    base = g.data_start + t.offset + e * rows * row_bytes
    return np.asarray(mm[base: base + rows * row_bytes])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="shard 1 of the file with the integrated draft block")
    ap.add_argument("--out", required=True, help="runtime dir to write (experts.bin, dense.bin, dense.txt)")
    ap.add_argument("--blk", type=int, default=48, help="draft block number (default 48)")
    ap.add_argument("--swap-eh", action="store_true",
                    help="split eh_proj as hid,emb instead of emb,hid (see docstring)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = f"blk.{a.blk}."
    model = Model(Path(a.gguf))
    log = []

    def src_bytes(name: str, want_type: str | None = None) -> bytes:
        g, t, mm, p = model.where[B + name]
        if want_type is not None:
            assert t.type_name == want_type, (name, t.type_name, want_type)
        n = t.expected_bytes()
        b = bytes(np.asarray(mm[g.data_start + t.offset: g.data_start + t.offset + n]))
        log.append({"tensor": B + name, "shard": p.name,
                    "source_sha256": hashlib.sha256(b).hexdigest()})
        return b

    # ---- experts: fuse gate/up along rows (lossless), requant to the draft Q2_0 grid ----
    with open(out / "experts.bin", "wb") as f:
        for e in range(NE):
            gate = q8_rows(expert_bytes(model, B + "ffn_gate_exps.weight", e, FF, H), FF, H)
            up = q8_rows(expert_bytes(model, B + "ffn_up_exps.weight", e, FF, H), FF, H)
            dn = q8_rows(expert_bytes(model, B + "ffn_down_exps.weight", e, H, FF), H, FF)
            gu = q2_0(np.concatenate([gate, up], axis=0)).reshape(2 * FF, H // 64 * 18)
            dq = q2_0(dn).reshape(H, FF // 64 * 18)
            f.write(blob_of(gu, dq))
    print(f"experts.bin {NE} blobs from {B}ffn_*_exps (Q8_0 -> draft Q2_0)", flush=True)

    # ---- dense ----
    lines, off = [], 0
    with open(out / "dense.bin", "wb") as f:
        def emit(short: str, kind: str, rows: int, cols: int, raw: bytes):
            nonlocal off
            want = {"q8_0": rows * (cols // 32) * Q8_ROW,
                    "bf16": rows * cols * 2, "f32": rows * cols * 4}[kind]
            assert len(raw) == want, (short, len(raw), want)
            pad = (-off) % 256
            f.write(b"\0" * pad)
            off += pad
            f.write(raw)
            lines.append(f"{short} {kind} {rows} {cols} {off} {len(raw)}")
            off += len(raw)

        def emit_q8(short: str, gguf_name: str, rows: int, cols: int):
            """Q8_0 reused byte for byte (rows of cols//32 blocks)."""
            raw = src_bytes(gguf_name)
            n = rows * (cols // 32) * Q8_ROW
            assert len(raw) == n, (gguf_name, len(raw), n)
            emit(short, "q8_0", rows, cols, raw)

        def emit_bf16_from_q8(short: str, gguf_name: str):
            arr = np.frombuffer(src_bytes(gguf_name, "Q8_0"), dtype=np.uint8)
            raw = bf16_bytes(arr, "Q8_0")
            rows, cols = bf16_shape_of(gguf_name)
            assert len(raw) == rows * cols * 2, (short, len(raw))
            emit(short, "bf16", rows, cols, raw)

        def emit_f32_norm(short: str, gguf_name: str, n: int):
            arr = np.frombuffer(src_bytes(gguf_name, "F32"), dtype=np.float32).reshape(-1)
            assert arr.size == n, (gguf_name, arr.size, n)
            emit(short, "f32", 1, n, (arr + 1.0).astype(np.float32).tobytes())

        # attention + shared expert + indexer (Q8_0 kept)
        emit_q8("self_attn.q_proj.weight", "attn_q.weight", 12288, H)
        emit_q8("self_attn.k_proj.weight", "attn_k.weight", 512, H)
        emit_q8("self_attn.v_proj.weight", "attn_v.weight", 512, H)
        emit_q8("self_attn.o_proj.weight", "attn_output.weight", H, 6144)
        emit_q8("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight", FF, H)
        emit_q8("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight", FF, H)
        emit_q8("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight", H, FF)
        # fused indexer qk: q rows ++ k rows (lossless concat, same row width)
        q = src_bytes("indexer.q_proj.weight", "Q8_0")
        k = src_bytes("indexer.k_proj.weight", "Q8_0")
        assert len(q) == 512 * (H // 32) * Q8_ROW and len(k) == 128 * (H // 32) * Q8_ROW
        emit("self_attn.indexer.index_qk_proj.weight", "q8_0", 640, H, q + k)
        # eh_proj split along columns: [2560 rows x 160 blocks] -> 2 x [2560 x 80 blocks]
        eh = src_bytes("nextn.eh_proj.weight", "Q8_0")
        assert len(eh) == H * 160 * Q8_ROW, len(eh)
        rows = np.frombuffer(eh, dtype=np.uint8).reshape(H, 160, Q8_ROW)
        halves = (rows[:, 80:, :], rows[:, :80, :]) if a.swap_eh else (rows[:, :80, :], rows[:, 80:, :])
        emit("fc_embedding.weight", "q8_0", H, H, halves[0].tobytes())
        emit("fc_hidden.weight", "q8_0", H, H, halves[1].tobytes())
        # hyper-connections + mixer (Q8_0 -> BF16) and router (F32 -> BF16)
        for short, gguf_name in [
                ("attn_hyper_connection.block_inject_weight.weight", "hc_attn_inject.weight"),
                ("attn_hyper_connection.input_mix_weight_down.weight", "hc_attn_down.weight"),
                ("attn_hyper_connection.input_mix_weight_up.weight", "hc_attn_up.weight"),
                ("mlp_hyper_connection.block_inject_weight.weight", "hc_ffn_inject.weight"),
                ("mlp_hyper_connection.input_mix_weight_down.weight", "hc_ffn_down.weight"),
                ("mlp_hyper_connection.input_mix_weight_up.weight", "hc_ffn_up.weight"),
                ("hyper_connection_mixer.input_mix_weight_down.weight", "nextn.hc_head_down.weight"),
                ("hyper_connection_mixer.input_mix_weight_up.weight", "nextn.hc_head_up.weight")]:
            emit_bf16_from_q8(short, gguf_name)
        router = np.frombuffer(src_bytes("ffn_gate_inp.weight", "F32"), dtype=np.float32).reshape(512, H)
        emit("mlp.gate.weight", "bf16", 512, H, bf16_bytes(router, "F32"))
        emit("mlp.shared_expert_gate.weight", "bf16", 1, H, bf16_bytes(
            np.frombuffer(src_bytes("ffn_gate_inp_shexp.weight", "F32"), dtype=np.float32), "F32"))
        # norms (F32, Gemma +1)
        for short, gguf_name, n in [
                ("attn_hyper_connection.hc_norm.weight", "hc_attn_norm.weight", 10240),
                ("mlp_hyper_connection.hc_norm.weight", "hc_ffn_norm.weight", 10240),
                ("hyper_connection_mixer.hc_norm.weight", "nextn.hc_head_norm.weight", 10240),
                ("self_attn.k_norm.weight", "attn_k_norm.weight", 256),
                ("self_attn.q_norm.weight", "attn_q_norm.weight", 256),
                ("self_attn.indexer.k_layernorm.weight", "indexer.k_norm.weight", 128),
                ("self_attn.indexer.q_layernorm.weight", "indexer.q_norm.weight", 128),
                ("pre_fc_norm_embedding.weight", "nextn.enorm.weight", H),
                ("pre_fc_norm_hidden.weight", "nextn.hnorm.weight", 10240)]:
            emit_f32_norm(short, gguf_name, n)
    (out / "dense.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "mtp_from_gguf.report.json").write_text(json.dumps(
        {"tool": "tools/mtp_from_gguf.py", "block": a.blk,
         "swap_eh": a.swap_eh, "tensors": log}, indent=1) + "\n")
    print(f"dense.bin {off} B, {len(lines)} tensors -> {out}")
    for l in lines:
        print("  " + l)
    return 0


def bf16_shape_of(gguf_name: str) -> tuple[int, int]:
    # rows/cols of the BF16 image are asserted by the engine against dense.txt; shapes here mirror mtp_rt's table.
    shapes = {
        "hc_attn_inject.weight": (4, 10240), "hc_attn_down.weight": (320, 10240),
        "hc_attn_up.weight": (10240, 320), "hc_ffn_inject.weight": (4, 10240),
        "hc_ffn_down.weight": (320, 10240), "hc_ffn_up.weight": (10240, 320),
        "nextn.hc_head_down.weight": (320, 10240), "nextn.hc_head_up.weight": (10240, 320),
    }
    return shapes[gguf_name]


if __name__ == "__main__":
    sys.exit(main())
