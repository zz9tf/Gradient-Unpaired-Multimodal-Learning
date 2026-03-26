import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional


# =========================================================
# pure helpers
# =========================================================
def zero_like(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_tensor(0.0)


def collect_by_slices(g: torch.Tensor, slices: List[Tuple[int, int]]) -> torch.Tensor:
    return torch.cat([g[s:e] for s, e in slices], dim=0)


def safe_cos(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.dot(a, b) / ((a.norm() + eps) * (b.norm() + eps))


def signed_js(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    a_pos = torch.clamp(a, min=0.0)
    a_neg = torch.clamp(-a, min=0.0)
    b_pos = torch.clamp(b, min=0.0)
    b_neg = torch.clamp(-b, min=0.0)

    qa = torch.cat([a_pos, a_neg], dim=0)
    qb = torch.cat([b_pos, b_neg], dim=0)

    qa = qa / (qa.sum() + eps)
    qb = qb / (qb.sum() + eps)
    m = 0.5 * (qa + qb)

    return 0.5 * (
        (qa * ((qa + eps).log() - (m + eps).log())).sum()
        + (qb * ((qb + eps).log() - (m + eps).log())).sum()
    )


def mag_gap(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    return (a.abs() - b.abs()).abs().sum() / (a.abs().sum() + b.abs().sum() + eps)


def conflict_mass(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    mass = a.abs() * b.abs()
    mask = (a * b) < 0
    return mass[mask].sum() / (mass.sum() + eps)


def sign_disagree(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    mask = (a.abs() > tau) & (b.abs() > tau)
    if not mask.any():
        return zero_like(a)
    return (torch.sign(a[mask]) != torch.sign(b[mask])).float().mean()


# =========================================================
# config
# =========================================================
@dataclass
class MonitorConfig:
    prefix: str = ""
    eps: float = 1e-8

    gpop_beta: float = 0.99

    enable_common_block: bool = True
    common_block_name: str = "__common__"

    relation_tau: float = 1e-8

    def validate(self):
        if not (0.0 < float(self.gpop_beta) < 1.0):
            raise ValueError(f"gpop_beta must be in (0,1), got {self.gpop_beta}")
        if float(self.eps) <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if float(self.relation_tau) < 0:
            raise ValueError(f"relation_tau must be >= 0, got {self.relation_tau}")


# =========================================================
# block
# =========================================================
class GradientBlockState:
    def __init__(
        self,
        name: str,
        slices: List[Tuple[int, int]],
        keys: List[str],
        cfg: MonitorConfig,
        ref_device: torch.device,
        ref_dtype: torch.dtype,
    ):
        self.name = name
        self.slices = list(slices)
        self.keys = list(keys)
        self.key_to_row = {k: i for i, k in enumerate(self.keys)}

        self.cfg = cfg
        self.ref_device = ref_device
        self.ref_dtype = ref_dtype

        self.pb = sum(e - s for s, e in self.slices)
        self.tb = len(self.keys)

        self.prev_gpop = torch.zeros(0, self.pb, device=ref_device, dtype=ref_dtype)
        self.gpop = torch.zeros(self.tb, self.pb, device=ref_device, dtype=ref_dtype)
        self.activated_rows = torch.zeros(self.tb, device=ref_device, dtype=torch.bool)   # [tb] bool

        self.g = torch.zeros(0, self.pb, device=ref_device, dtype=ref_dtype)
        self.current_keys: List[str] = []
        self.valid_mask = None

    @torch.no_grad()
    def build_current(
        self,
        G: torch.Tensor,          # [T_cur, P]
        loss_keys: List[str],
    ):
        g_list = []
        cur_keys = []
        valid_mask = torch.zeros(self.tb, device=G.device, dtype=torch.bool)

        for i, loss_key in enumerate(loss_keys):
            row = self.key_to_row.get(loss_key, None)
            if row is None:
                continue

            gi = collect_by_slices(G[i], self.slices)
            g_list.append(gi)
            cur_keys.append(loss_key)
            valid_mask[row] = True

        if len(g_list) > 0:
            self.g = torch.stack(g_list, dim=0)
        else:
            self.g = torch.zeros(0, self.pb, device=G.device, dtype=G.dtype)

        self.current_keys = cur_keys
        self.valid_mask = valid_mask

    @torch.no_grad()
    def update_gpop(self, step: int):
        self.prev_gpop = None if self.gpop is None else self.gpop.detach().clone()

        if len(self.current_keys) == 0:
            return

        for i, loss_key in enumerate(self.current_keys):
            row = self.key_to_row[loss_key]
            if not self.activated_rows[row]:
                self.gpop[row] = self.g[i].detach()
                self.activated_rows[row] = True
            else:
                beta = float(self.cfg.gpop_beta)
                src = self.g[i].detach()
                self.gpop[row] = beta * self.gpop[row] + (1.0 - beta) * src

    @torch.no_grad()
    def compute_stats(self) -> Dict[str, torch.Tensor]:
        eps = float(self.cfg.eps)
        tau = float(self.cfg.relation_tau)
        stats: Dict[str, torch.Tensor] = {}

        cur_key_to_i = {k: i for i, k in enumerate(self.current_keys)}

        for loss_key in self.keys:
            row = self.key_to_row[loss_key]
            if not self.activated_rows[row]:
                continue
                
            p = self.gpop[row]
            p_prev = self.prev_gpop[row]

            if loss_key in cur_key_to_i:
                g = self.g[cur_key_to_i[loss_key]]
                stats[f"{self.name}.{loss_key}.gpop_self_align"] = safe_cos(g, p, eps)
                stats[f"{self.name}.{loss_key}.gpop_norm_ratio"] = (p.norm() + eps) / (g.norm() + eps)
            else:
                z = zero_like(p)
                stats[f"{self.name}.{loss_key}.gpop_self_align"] = z
                stats[f"{self.name}.{loss_key}.gpop_norm_ratio"] = z

            stats[f"{self.name}.{loss_key}.gpop_drift"] = safe_cos(p, p_prev, eps)
            stats[f"{self.name}.{loss_key}.gpop_std"] = p.std(unbiased=False) if p.numel() > 0 else zero_like(p)
            stats[f"{self.name}.{loss_key}.gpop_pos_frac"] = (p > 0).float().mean() if p.numel() > 0 else zero_like(p)

        pair_signed_js = []
        pair_mag_gap = []
        pair_conflict = []
        pair_cos = []
        pair_sign_dis = []

        T = len(self.keys)
        for i in range(T):
            for j in range(i + 1, T):
                ki, kj = self.keys[i], self.keys[j]
                ri, rj = self.key_to_row[ki], self.key_to_row[kj]
                pi, pj = self.gpop[ri], self.gpop[rj]

                both_present = bool(self.valid_mask[ri].item() and self.valid_mask[rj].item())
                if both_present:
                    sjs = signed_js(pi, pj, eps)
                    mg = mag_gap(pi, pj, eps)
                    cm = conflict_mass(pi, pj, eps)
                    cs = safe_cos(pi, pj, eps)
                    sd = sign_disagree(pi, pj, tau)

                    pair_signed_js.append(sjs)
                    pair_mag_gap.append(mg)
                    pair_conflict.append(cm)
                    pair_cos.append(cs)
                    pair_sign_dis.append(sd)
                else:
                    nanv = torch.full((), float("nan"), device=pi.device, dtype=pi.dtype)
                    sjs = mg = cm = cs = sd = nanv

                stats[f"{self.name}.{ki}.{kj}.gpop_signed_js"] = sjs
                stats[f"{self.name}.{ki}.{kj}.gpop_mag_gap"] = mg
                stats[f"{self.name}.{ki}.{kj}.gpop_conflict_mass"] = cm
                stats[f"{self.name}.{ki}.{kj}.gpop_cos"] = cs
                stats[f"{self.name}.{ki}.{kj}.gpop_sign_disagree"] = sd

        if len(pair_signed_js) > 0:
            stats[f"{self.name}.gpop_pair_signed_js_mean"] = torch.stack(pair_signed_js).mean()
            stats[f"{self.name}.gpop_pair_mag_gap_mean"] = torch.stack(pair_mag_gap).mean()
            stats[f"{self.name}.gpop_pair_conflict_mass_mean"] = torch.stack(pair_conflict).mean()
            stats[f"{self.name}.gpop_pair_cos_mean"] = torch.stack(pair_cos).mean()
            stats[f"{self.name}.gpop_pair_sign_disagree_mean"] = torch.stack(pair_sign_dis).mean()
        else:
            ref = self.gpop[0] if self.gpop.shape[0] > 0 else torch.zeros((), device=self.ref_device, dtype=self.ref_dtype)
            z = zero_like(ref)
            stats[f"{self.name}.gpop_pair_signed_js_mean"] = z
            stats[f"{self.name}.gpop_pair_mag_gap_mean"] = z
            stats[f"{self.name}.gpop_pair_conflict_mass_mean"] = z
            stats[f"{self.name}.gpop_pair_cos_mean"] = z
            stats[f"{self.name}.gpop_pair_sign_disagree_mean"] = z

        return stats


# =========================================================
# monitor
# =========================================================
class GradientMonitor:
    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        block_split_fn: Callable[[str], str],
        block_loss_keys: Dict[str, List[str]],
        cfg: Optional[MonitorConfig] = None,
        common_slices: Optional[List[Tuple[int, int]]] = None,
    ):
        self.cfg = cfg or MonitorConfig()
        self.cfg.validate()

        self.common_slices = common_slices or []
        self.block_loss_keys = {k: list(v) for k, v in block_loss_keys.items()}

        self.param_slices = self._build_param_slices(named_params, block_split_fn)
        self._ref_device, self._ref_dtype = self._infer_ref_device_dtype(named_params)

        self.blocks: Dict[str, GradientBlockState] = {}
        self._init_blocks()

        self._step = 0

    def _infer_ref_device_dtype(self, named_params):
        for _, p in named_params:
            return p.device, p.dtype
        return torch.device("cpu"), torch.float32

    def _build_param_slices(self, named_params, block_split_fn):
        param_slices = {}
        offset = 0
        for name, p in named_params:
            if not p.requires_grad:
                continue
            block = block_split_fn(name)
            n = p.numel()
            param_slices.setdefault(block, []).append((offset, offset + n))
            offset += n
        return param_slices

    def _init_blocks(self):
        for block_name, slices in self.param_slices.items():
            if block_name not in self.block_loss_keys:
                raise ValueError(f"Missing loss schema for block: {block_name}")
            self.blocks[block_name] = GradientBlockState(
                name=block_name,
                slices=slices,
                keys=self.block_loss_keys[block_name],
                cfg=self.cfg,
                ref_device=self._ref_device,
                ref_dtype=self._ref_dtype,
            )

        if self.cfg.enable_common_block and len(self.common_slices) > 0:
            block_name = self.cfg.common_block_name
            if block_name not in self.block_loss_keys:
                raise ValueError(f"Missing loss schema for block: {block_name}")
            self.blocks[block_name] = GradientBlockState(
                name=block_name,
                slices=self.common_slices,
                keys=self.block_loss_keys[block_name],
                cfg=self.cfg,
                ref_device=self._ref_device,
                ref_dtype=self._ref_dtype,
            )

    def _apply_prefix(self, stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = (self.cfg.prefix or "").strip()
        if p == "":
            return stats
        if not p.endswith("."):
            p = p + "."
        return {p + k: v for k, v in stats.items()}

    @torch.no_grad()
    def monitor(
        self,
        G: torch.Tensor,
        loss_keys: List[str],
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if G.ndim != 2:
            raise ValueError(f"G must be 2D [T, P], got shape={tuple(G.shape)}")
        if G.shape[0] != len(loss_keys):
            raise ValueError(f"G.shape[0] ({G.shape[0]}) != len(loss_keys) ({len(loss_keys)})")

        if step is None:
            self._step += 1
        else:
            self._step = int(step)

        stats: Dict[str, torch.Tensor] = {
            "monitor.step": G.new_tensor(float(self._step)),
            "monitor.num_losses_cur": G.new_tensor(float(G.shape[0])),
        }

        for block in self.blocks.values():
            block.build_current(G=G, loss_keys=loss_keys)
            block.update_gpop(step=self._step)
            if block.prev_gpop is not None:
                stats.update(block.compute_stats())

        return self._apply_prefix(stats)