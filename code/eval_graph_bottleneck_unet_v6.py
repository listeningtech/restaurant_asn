#!/usr/bin/env python3
"""
eval_graph_bottleneck_unet_v6.py

Evaluate GraphBottleneckUNetV6 on RestaurantSim.

Supports the clean comparison introduced in v6:
- Graph ON : cross-node graph fusion at bottleneck
- Graph OFF: strict local-only baseline

The script restores architecture settings from checkpoint args when available.
It reports:
- SI-SDR_in  : ref-mic mixture vs target
- SI-SDR_out : model output vs target
- Delta      : SI-SDR_out - SI-SDR_in

Optional:
- save a few scene examples as wav files
- full-length evaluation with --full_length 1

Example:
  CUDA_VISIBLE_DEVICES=0 python eval_graph_bottleneck_unet_v6.py     --data /home/rrame12/Desktop/Datasets/RestaurantSim_v4_m1_varspk     --ckpt /home/rrame12/Desktop/Research/ASN/runs_graph_bottleneck_unet_v6_on/best.pt     --out  /home/rrame12/Desktop/Research/ASN/runs_graph_bottleneck_unet_v6_on/eval_test     --batch 4 --num_workers 6 --full_length 1 --save_scenes 6
"""

import os
import glob
import json
import argparse
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm

SR = 16000
EPS = 1e-8


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target = ((est * ref).sum(dim=-1, keepdim=True) / ref_energy) * ref
    e_noise = est - s_target
    ratio = (s_target * s_target).sum(dim=-1) / ((e_noise * e_noise).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def read_dataset_manifest(data_root: str) -> Dict:
    manifest_path = os.path.join(data_root, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found at: {manifest_path}")
    return read_json(manifest_path)


def resolve_n_mics(data_root: str, ckpt_args: Dict, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    if "resolved_n_mics" in ckpt_args:
        return int(ckpt_args["resolved_n_mics"])
    if "n_mics_override" in ckpt_args and ckpt_args["n_mics_override"] is not None:
        return int(ckpt_args["n_mics_override"])
    manifest = read_dataset_manifest(data_root)
    if "M" not in manifest:
        raise KeyError(f"'M' not found in {os.path.join(data_root, 'manifest.json')}")
    return int(manifest["M"])


class RestaurantSceneDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str, crop_samples: Optional[int], cache: bool = True):
        super().__init__()
        self.crop_samples = crop_samples
        self.cache = cache
        self.shard_paths = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))
        if not self.shard_paths:
            raise RuntimeError(f"No shards found in {split_dir}")

        self.index: List[Tuple[int, int]] = []
        self._sizes: List[int] = []
        for sp in self.shard_paths:
            with np.load(sp) as d:
                B = d["Y"].shape[0]
            self._sizes.append(B)
        for sid, B in enumerate(self._sizes):
            for i in range(B):
                self.index.append((sid, i))

        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    def __len__(self):
        return len(self.index)

    def _load(self, sid: int) -> Dict[str, np.ndarray]:
        if self.cache and sid in self._cache:
            return self._cache[sid]
        d = np.load(self.shard_paths[sid])
        sh = {k: d[k] for k in d.files}
        d.close()
        if self.cache:
            self._cache[sid] = sh
        return sh

    def __getitem__(self, idx: int):
        sid, li = self.index[idx]
        sh = self._load(sid)
        Y = sh["Y"][li]                  # (K,M,T)
        S = sh["target_refclean"][li]    # (K,T)
        A = sh["adj"][li].astype(np.float32)

        K, M, TT = Y.shape
        if self.crop_samples is not None and self.crop_samples < TT:
            start = np.random.randint(0, TT - self.crop_samples + 1)
            Y = Y[:, :, start:start + self.crop_samples]
            S = S[:, start:start + self.crop_samples]

        return {
            "Y": torch.from_numpy(Y).float(),
            "S": torch.from_numpy(S).float(),
            "A": torch.from_numpy(A).float(),
        }


def collate_fn(batch):
    return {
        "Y": torch.stack([b["Y"] for b in batch], 0),
        "S": torch.stack([b["S"] for b in batch], 0),
        "A": torch.stack([b["A"] for b in batch], 0),
    }


class TorchSTFT(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 128, win_length: Optional[int] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win_length = win_length if win_length is not None else n_fft
        self.register_buffer("window", torch.hann_window(self.win_length))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        N, M, T = x.shape
        x = x.reshape(N * M, T)
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        Freq, Frames = X.shape[-2], X.shape[-1]
        return X.reshape(N, M, Freq, Frames)

    def istft(self, X: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=length,
        )


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1, drop: float = 0.0):
        super().__init__()
        ng = min(8, cout)
        while cout % ng != 0 and ng > 1:
            ng -= 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(ng, cout),
            nn.PReLU(),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, drop: float = 0.0):
        super().__init__()
        self.c1 = ConvGNAct(cin, cout, drop=drop)
        self.c2 = ConvGNAct(cout, cout, drop=drop)
        self.down = nn.Conv2d(cout, cout, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.c2(self.c1(x))
        skip = x
        x = self.down(x)
        return skip, x


class UpBlock(nn.Module):
    def __init__(self, cin: int, skip_ch: int, cout: int, drop: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.c1 = ConvGNAct(cout + skip_ch, cout, drop=drop)
        self.c2 = ConvGNAct(cout, cout, drop=drop)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh != 0 or dw != 0:
            x = F.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
            x = x[..., :skip.shape[-2], :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.c2(self.c1(x))


class MeanPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.out_dim = in_ch
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=(-2, -1))


class MeanStdPool(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.out_dim = 2 * in_ch
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mu = h.mean(dim=(-2, -1))
        std = h.std(dim=(-2, -1), unbiased=False)
        return torch.cat([mu, std], dim=1)


class AttnPool(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.PReLU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.out_dim = in_ch
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        BK, C, Freq, Frames = h.shape
        s = self.score(h).view(BK, -1)
        a = torch.softmax(s, dim=-1).view(BK, 1, Freq, Frames)
        return (a * h).sum(dim=(-2, -1))


def build_pool(pool_name: str, bott_ch: int) -> nn.Module:
    pool_name = pool_name.lower()
    if pool_name == "mean":
        return MeanPool(bott_ch)
    if pool_name == "meanstd":
        return MeanStdPool(bott_ch)
    if pool_name == "attn":
        return AttnPool(bott_ch)
    raise ValueError(f"Unknown pool type: {pool_name}")


class GATLayer(nn.Module):
    def __init__(self, din: int, dout: int, heads: int = 2, dropout: float = 0.0, leaky: float = 0.2):
        super().__init__()
        self.dout = dout
        self.heads = heads
        self.dropout = dropout
        self.W = nn.Linear(din, heads * dout, bias=False)
        self.a = nn.Parameter(torch.randn(heads, 2 * dout) * 0.02)
        self.bias = nn.Parameter(torch.zeros(heads * dout))
        self.leaky = leaky

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, K, _ = z.shape
        A = A.clone()
        eye = torch.eye(K, device=A.device, dtype=A.dtype)[None, :, :]
        A = torch.clamp(A + eye, 0.0, 1.0)

        Wh = self.W(z).view(B, K, self.heads, self.dout)
        Wh_i = Wh[:, :, None, :, :].expand(B, K, K, self.heads, self.dout)
        Wh_j = Wh[:, None, :, :, :].expand(B, K, K, self.heads, self.dout)
        cat = torch.cat([Wh_i, Wh_j], dim=-1)

        e = (cat * self.a[None, None, None, :, :]).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.leaky)

        mask = (A[:, :, :, None] > 0.0)
        e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, dim=2)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = (alpha[..., None] * Wh_j).sum(dim=2)
        out = out.reshape(B, K, self.heads * self.dout) + self.bias
        return out


class GATStack(nn.Module):
    def __init__(self, din: int, dhid: int, dout: int, layers: int = 2, heads: int = 2, dropout: float = 0.0):
        super().__init__()
        mods = []
        in_d = din
        for li in range(layers):
            out_d = dout if li == layers - 1 else dhid
            mods.append(GATLayer(in_d, out_d, heads=heads, dropout=dropout))
            in_d = heads * out_d
        self.mods = nn.ModuleList(mods)

    def forward(self, z: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = z
        for i, m in enumerate(self.mods):
            h = m(h, A)
            if i != len(self.mods) - 1:
                h = F.elu(h)
        return h


class GraphBottleneckUNetV6(nn.Module):
    def __init__(
        self,
        n_mics: int,
        n_fft: int = 512,
        hop: int = 128,
        base: int = 32,
        depth: int = 4,
        drop: float = 0.0,
        graph_dim: int = 128,
        graph_layers: int = 2,
        graph_heads: int = 2,
        pool: str = "mean",
        use_graph: bool = True,
        alpha_init: float = 1.0,
        graph_dropout: float = 0.0,
    ):
        super().__init__()
        assert depth >= 2
        self.n_mics = int(n_mics)
        self.use_graph = bool(use_graph)
        self.stft = TorchSTFT(n_fft=n_fft, hop=hop)

        in_ch = 3 * self.n_mics
        enc = []
        chs = []
        c_in = in_ch
        c = base
        for _ in range(depth):
            enc.append(DownBlock(c_in, c, drop=drop))
            chs.append(c)
            c_in = c
            c *= 2
        self.enc = nn.ModuleList(enc)

        bott_ch = chs[-1] * 2
        self.bot1 = ConvGNAct(chs[-1], bott_ch, drop=drop)
        self.bot2 = ConvGNAct(bott_ch, bott_ch, drop=drop)

        self.pool = build_pool(pool, bott_ch)
        pool_dim = self.pool.out_dim
        self.pre_graph = nn.Sequential(nn.Linear(pool_dim, graph_dim), nn.PReLU())
        self.gnn = GATStack(din=graph_dim, dhid=graph_dim, dout=graph_dim,
                            layers=graph_layers, heads=graph_heads, dropout=graph_dropout)
        self.post_graph = nn.Sequential(
            nn.Linear(graph_heads * graph_dim, graph_dim),
            nn.PReLU(),
            nn.Linear(graph_dim, bott_ch),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

        dec = []
        c_in = bott_ch
        for i in reversed(range(depth)):
            dec.append(UpBlock(c_in, chs[i], chs[i], drop=drop))
            c_in = chs[i]
        self.dec = nn.ModuleList(dec)
        self.out_mask = nn.Conv2d(base, 1, kernel_size=1)

    def _make_features(self, X: torch.Tensor):
        mag = torch.abs(X).clamp_min(EPS)
        logmag = torch.log1p(mag)
        real = X.real / (mag + EPS)
        imag = X.imag / (mag + EPS)
        feat = torch.cat([logmag, real, imag], dim=1)
        X_ref = X[:, 0]
        return feat, X_ref

    def _encode(self, feat: torch.Tensor):
        skips = []
        x = feat
        for blk in self.enc:
            s, x = blk(x)
            skips.append(s)
        x = self.bot2(self.bot1(x))
        return skips, x

    def _apply_graph(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, K, C, Freq, Frames = h.shape
        hk = h.reshape(B * K, C, Freq, Frames)
        z = self.pool(hk).reshape(B, K, -1)
        z = self.pre_graph(z)
        z = self.gnn(z, A)
        z = self.post_graph(z)
        delta = z[:, :, :, None, None].expand(B, K, C, Freq, Frames)
        return h + self.alpha * delta

    def _decode(self, h: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = h
        for blk, skip in zip(self.dec, reversed(skips)):
            x = blk(x, skip)
        return torch.sigmoid(self.out_mask(x).squeeze(1))

    def forward(self, Y: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, M, T = Y.shape
        assert M == self.n_mics, f"Expected n_mics={self.n_mics}, got {M}"
        Yflat = Y.reshape(B * K, M, T)
        X = self.stft.stft(Yflat)
        feat, X_ref = self._make_features(X)
        skips, h = self._encode(feat)
        _, C, Freq, Frames = h.shape
        if self.use_graph:
            if A is None:
                raise ValueError("Adjacency A must be provided when graph is enabled.")
            h_nodes = h.reshape(B, K, C, Freq, Frames)
            h_nodes = self._apply_graph(h_nodes, A)
            h = h_nodes.reshape(B * K, C, Freq, Frames)
        mask = self._decode(h, skips)
        Yhat_spec = mask * X_ref
        yhat = self.stft.istft(Yhat_spec, length=T).reshape(B, K, T)
        return yhat


def restore_model_from_ckpt(ckpt: Dict, data_root: str, n_mics_override: Optional[int] = None) -> Tuple[nn.Module, Dict]:
    ckpt_args = ckpt.get("args", {})
    n_mics = resolve_n_mics(data_root, ckpt_args, override=n_mics_override)
    model = GraphBottleneckUNetV6(
        n_mics=n_mics,
        n_fft=int(ckpt_args.get("nfft", 512)),
        hop=int(ckpt_args.get("hop", 128)),
        base=int(ckpt_args.get("base", 32)),
        depth=int(ckpt_args.get("depth", 4)),
        drop=float(ckpt_args.get("drop", 0.0)),
        graph_dim=int(ckpt_args.get("graph_dim", 128)),
        graph_layers=int(ckpt_args.get("graph_layers", 2)),
        graph_heads=int(ckpt_args.get("graph_heads", 2)),
        pool=str(ckpt_args.get("pool", "mean")),
        use_graph=(not bool(ckpt_args.get("disable_graph", False))),
        alpha_init=float(ckpt_args.get("alpha_init", 1.0)),
        graph_dropout=float(ckpt_args.get("graph_dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    return model, ckpt_args


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, out_dir: str, save_scenes: int = 0) -> Dict:
    model.eval()
    ensure_dir(out_dir)
    audio_dir = os.path.join(out_dir, "audio_examples")
    if save_scenes > 0:
        ensure_dir(audio_dir)

    total_in = 0.0
    total_out = 0.0
    total_n = 0
    saved = 0
    per_scene = []

    pbar = tqdm(loader, desc="Evaluating", ncols=110)
    for batch_idx, batch in enumerate(pbar):
        Y = batch["Y"].to(device, non_blocking=True)
        S = batch["S"].to(device, non_blocking=True)
        A = batch["A"].to(device, non_blocking=True)

        Yhat = model(Y, A)
        sisdr_out = compute_si_sdr(Yhat, S)          # (B,K)
        sisdr_in = compute_si_sdr(Y[:, :, 0, :], S)  # (B,K)
        delta = sisdr_out - sisdr_in

        B, K, T = Yhat.shape
        total_in += sisdr_in.sum().item()
        total_out += sisdr_out.sum().item()
        total_n += B * K

        mean_delta = delta.mean().item()
        pbar.set_postfix({"delta": f"{mean_delta:+.2f} dB"})

        for b in range(B):
            for k in range(K):
                per_scene.append({
                    "batch_index": int(batch_idx),
                    "sample_in_batch": int(b),
                    "table": int(k),
                    "sisdr_in_db": float(sisdr_in[b, k].item()),
                    "sisdr_out_db": float(sisdr_out[b, k].item()),
                    "delta_db": float(delta[b, k].item()),
                })

        if saved < save_scenes:
            n_to_save = min(B, save_scenes - saved)
            Y_cpu = Y.cpu().numpy()
            S_cpu = S.cpu().numpy()
            Yhat_cpu = Yhat.cpu().numpy()
            for b in range(n_to_save):
                scene_dir = os.path.join(audio_dir, f"scene_{saved:03d}")
                ensure_dir(scene_dir)
                for k in range(K):
                    sf.write(os.path.join(scene_dir, f"table{k}_mix_ref.wav"), Y_cpu[b, k, 0], SR)
                    sf.write(os.path.join(scene_dir, f"table{k}_target.wav"), S_cpu[b, k], SR)
                    sf.write(os.path.join(scene_dir, f"table{k}_enh.wav"), Yhat_cpu[b, k], SR)
                saved += 1
                if saved >= save_scenes:
                    break

    summary = {
        "sisdr_in_db": total_in / max(total_n, 1),
        "sisdr_out_db": total_out / max(total_n, 1),
        "delta_db": (total_out - total_in) / max(total_n, 1),
        "n_examples": int(total_n),
    }
    write_json(os.path.join(out_dir, "metrics_summary.json"), summary)
    write_json(os.path.join(out_dir, "metrics_per_scene.json"), {"rows": per_scene})
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--full_length", type=int, default=1, help="1 => evaluate full length, 0 => use crop_s from ckpt args")
    ap.add_argument("--crop_s", type=float, default=None, help="optional override if full_length=0")
    ap.add_argument("--save_scenes", type=int, default=0)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--n_mics_override", type=int, default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model, ckpt_args = restore_model_from_ckpt(ckpt, args.data, n_mics_override=args.n_mics_override)
    model = model.to(device)

    if args.full_length:
        crop_samples = None
    else:
        if args.crop_s is not None:
            crop_s = args.crop_s
        else:
            crop_s = float(ckpt_args.get("crop_s", 2.0))
        crop_samples = None if crop_s <= 0 else int(round(crop_s * SR))

    split_dir = os.path.join(args.data, args.split)
    ds = RestaurantSceneDataset(split_dir=split_dir, crop_samples=crop_samples, cache=True)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )

    print("=" * 80)
    print("Evaluate Graph Bottleneck U-Net v6")
    print(f"device          : {device}")
    print(f"data            : {args.data}")
    print(f"split           : {args.split}")
    print(f"ckpt            : {args.ckpt}")
    print(f"out             : {args.out}")
    print(f"graph_enabled   : {not bool(ckpt_args.get('disable_graph', False))}")
    print(f"resolved_n_mics : {resolve_n_mics(args.data, ckpt_args, override=args.n_mics_override)}")
    print(f"crop_samples    : {crop_samples}")
    print("=" * 80)

    summary = evaluate(model, loader, device, args.out, save_scenes=args.save_scenes)

    print("\n=== TEST RESULTS ===")
    print(f"mode: {'GRAPH_ON' if not bool(ckpt_args.get('disable_graph', False)) else 'GRAPH_OFF'}")
    print(f"ckpt: {args.ckpt}")
    print(f"SI-SDR_in : {summary['sisdr_in_db']:.2f} dB")
    print(f"SI-SDR_out: {summary['sisdr_out_db']:.2f} dB")
    print(f"Delta     : {summary['delta_db']:+.2f} dB")


if __name__ == "__main__":
    main()