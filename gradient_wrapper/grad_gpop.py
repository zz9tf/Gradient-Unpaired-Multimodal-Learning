from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# =========================================================
# Utils
# =========================================================

def _as_tensor_dict(stats: Dict, device, dtype) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            out[k] = v.detach()
        else:
            out[k] = torch.tensor(float(v), device=device, dtype=dtype)
    return out


def _prefix(stats: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {f"{prefix}.{k}": v for k, v in stats.items()}


def build_common_ids(
    named_params: List[Tuple[str, torch.nn.Parameter]],
    common_param_filter: Callable[[str], bool],
) -> torch.Tensor:
    col_ids = []
    offset = 0
    for name, p in named_params:
        if not p.requires_grad:
            continue

        n = p.numel()
        if bool(common_param_filter(name)):
            col_ids.append(torch.arange(offset, offset + n, dtype=torch.long))
        offset += n

    return torch.cat(col_ids, dim=0) if col_ids else torch.empty(0, dtype=torch.long)


def _safe_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.linalg.norm(x).clamp_min(eps)


# =========================================================
# Config
# =========================================================

@dataclass
class CommonGpopConfig:
    gpop_keys: List[str] = field(default_factory=list)
    ref_build_kind: str = "cov"   # "gg" or "cov" or "weighted_mean"
    unbiased: bool = True
    cov_center: bool = True

    damping: float = 1e-3
    cg_max_iter: int = 30
    cg_tol: float = 1e-6

    ema_beta: float = 0.999
    eps: float = 1e-8
    edit_kind: str = "project"   # "project"

    def validate(self):
        if self.ref_build_kind not in ("gg", "cov", "weighted_mean"):
            raise ValueError(f"[gpop] operator_kind must be one of ['gg','cov','weighted_mean'], got {self.ref_build_kind}")
        if float(self.damping) < 0:
            raise ValueError(f"[gpop] damping must be >= 0, got {self.damping}")
        if int(self.cg_max_iter) <= 0:
            raise ValueError(f"[gpop] cg_max_iter must be > 0, got {self.cg_max_iter}")
        if float(self.cg_tol) <= 0:
            raise ValueError(f"[gpop] cg_tol must be > 0, got {self.cg_tol}")
        if not (0.0 < float(self.ema_beta) < 1.0):
            raise ValueError(f"[gpop] ema_beta must be in (0,1), got {self.ema_beta}")
        if float(self.eps) <= 0:
            raise ValueError(f"[gpop] eps must be > 0, got {self.eps}")


# =========================================================
# EMA state
# =========================================================

class CommonGpopState:
    """
    State only owns:
      - schema
      - EMA memory
      - current batch cache

    Internal semantics:
      schema: fixed key -> row mapping
      ref_G: [Tb, Pc], fixed rows under schema
      cur_schema_ids: [T], current valid schema ids
      cur_G: [T, Pc], current valid rows only
      cur_keys: [T], current valid keys aligned with cur_G
      cur_src_ids: [T], current valid source ids
      activated_rows: [Tb] bool, activated schema ids
    """

    def __init__(self, ema_beta: float, gpop_keys: List[str]):
        self.ema_beta = float(ema_beta)

        # fixed schema
        self.schema: Dict[str, int] = {key: i for i, key in enumerate(gpop_keys)}

        # EMA memory under schema
        self.ref_G: Optional[torch.Tensor] = None          # [Tb, Pc]

        # current batch cache
        self.cur_G: Optional[torch.Tensor] = None          # [T, Pc]
        self.cur_keys: List[str] = []                      # len = T
        self.cur_schema_ids: Optional[torch.Tensor] = None    # [T], schema row ids
        self.cur_src_ids: Optional[torch.Tensor] = None      # [T], source ids
        
        self.activated_rows: Optional[torch.Tensor] = None   # [Tb] bool

    @property
    def num_schema(self) -> int:
        return len(self.schema)
    
    @torch.no_grad()
    def build_current(
        self,
        G_common: torch.Tensor,
        loss_keys: List[str],
        weights: Optional[torch.Tensor] = None,
    ):
        """
        Build current valid rows from current batch according to fixed schema.
        """
        if G_common.ndim != 2:
            raise ValueError(f"[gpop] G_common must be [T,Pc], got {tuple(G_common.shape)}")
        if G_common.shape[0] != len(loss_keys):
            raise ValueError(
                f"[gpop] G_common.shape[0] ({G_common.shape[0]}) != len(loss_keys) ({len(loss_keys)})"
            )

        device = G_common.device
        schema_ids: List[int] = []
        src_ids: List[int] = []
        cur_keys: List[str] = []

        for src_i, key in enumerate(loss_keys):
            if key not in self.schema:
                raise ValueError(f"[gpop] key {key} not found in schema")
            src_ids.append(src_i)
            schema_ids.append(self.schema[key])
            cur_keys.append(key)

        if len(src_ids) == 0:
            self.cur_G = G_common.new_zeros((0, G_common.shape[1]))
            self.cur_keys = []
            self.cur_schema_ids = torch.empty((0,), device=device, dtype=torch.long)
            self.cur_src_ids = torch.empty((0,), device=device, dtype=torch.long)
            return

        src_ids_t = torch.tensor(src_ids, device=device, dtype=torch.long)
        schema_ids_t = torch.tensor(schema_ids, device=device, dtype=torch.long)
        if schema_ids_t.numel() > 0 and schema_ids_t.numel() != torch.unique(schema_ids_t).numel():
            raise ValueError("[gpop] duplicate schema ids found in current batch")

        self.cur_G = G_common[src_ids_t].detach()
        if weights is not None:
            cur_weights = weights[schema_ids_t]
            self.cur_G = self.cur_G * cur_weights.unsqueeze(1)
        self.cur_keys = cur_keys
        self.cur_schema_ids = schema_ids_t
        self.cur_src_ids = src_ids_t

    @torch.no_grad()
    def update_ema(self):
        if self.cur_G is None or self.cur_G.shape[0] == 0 or self.cur_schema_ids is None:
            raise RuntimeError("[gpop] cur_G is not built")

        if self.ref_G is None:
            self.ref_G = torch.zeros(
                (self.num_schema, self.cur_G.shape[1]),
                device=self.cur_G.device,
                dtype=self.cur_G.dtype,
            )

        if self.activated_rows is None:
            self.activated_rows = torch.zeros(
                (self.num_schema,),
                device=self.cur_G.device,
                dtype=torch.bool,
            )

        is_active = self.activated_rows[self.cur_schema_ids]   # [T]

        init_schema_ids = self.cur_schema_ids[~is_active]
        update_schema_ids = self.cur_schema_ids[is_active]

        init_G = self.cur_G[~is_active]
        update_G = self.cur_G[is_active]

        if init_schema_ids.numel() > 0:
            self.ref_G[init_schema_ids] = init_G
            self.activated_rows[init_schema_ids] = True

        if update_schema_ids.numel() > 0:
            self.ref_G[update_schema_ids] = (
                self.ema_beta * self.ref_G[update_schema_ids]
                + (1.0 - self.ema_beta) * update_G
            )
            
    def state_dict(self) -> Dict:
        return {
            "ema_beta": self.ema_beta,
            "schema": self.schema,
            "ema_G": self.ref_G,
            "activated_rows": self.activated_rows,
        }

    def load_state_dict(self, sd: Dict):
        self.ema_beta = float(sd.get("ema_beta", self.ema_beta))
        self.schema = sd.get("schema", self.schema)
        self.ref_G = sd.get("ema_G", None)
        self.activated_rows = sd.get("activated_rows", None)

# =========================================================
# Reference builder
# =========================================================

class GpopRefBuilder:
    def __init__(self, cfg: CommonGpopConfig):
        self.cfg = cfg

    @torch.no_grad()
    def build_g_ref(
        self,
        G_cur: torch.Tensor,
        G_ref: torch.Tensor,
        activated_rows: torch.Tensor,
        gpop_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if G_cur.ndim != 2:
            raise ValueError(f"[gpop] G_common must be [T,Pc], got {tuple(G_cur.shape)}")
        if G_ref.ndim != 2:
            raise ValueError(f"[gpop] G_base must be [Tb, Pc], got {tuple(G_ref.shape)}")
        if G_cur.shape[1] != G_ref.shape[1]:
            raise ValueError(
                f"[gpop] mismatch: G_common={tuple(G_cur.shape)}, G_base={tuple(G_ref.shape)}"
            )
        
        T = int(G_cur.shape[0])
        denom = float(max(T - 1 if bool(self.cfg.unbiased) else T, 1))
        if activated_rows is None or not bool(activated_rows.any()):
            raise RuntimeError("[gpop] no activated rows in G_ref")
        G_ref_active = G_ref[activated_rows]
            
        if gpop_weights is None:
            g_ref = G_ref_active.mean(dim=0)
        else: 
            w_active = gpop_weights[activated_rows]
            g_ref = (G_ref_active * w_active.unsqueeze(1)).sum(dim=0) / w_active.sum().clamp_min(self.cfg.eps)

        if self.cfg.ref_build_kind == "weighted_mean":
            return g_ref

        if self.cfg.ref_build_kind == "gg":
            return self._gg_mul(G_cur, g_ref, denom)

        if self.cfg.ref_build_kind == "cov":
            return self._cov_inv_cg(
                X=G_cur,
                b=g_ref,
                damping=float(self.cfg.damping),
                max_iter=int(self.cfg.cg_max_iter),
                tol=float(self.cfg.cg_tol),
                eps=float(self.cfg.eps),
            )

        raise ValueError(f"[gpop] unsupported operator_kind: {self.cfg.ref_build_kind}")

    @torch.no_grad()
    def _gg_mul(
        self,
        X: torch.Tensor,
        v: torch.Tensor,
        denom: float,
    ) -> torch.Tensor:
        return (X.T @ (X @ v)) / denom

    @torch.no_grad()
    def _cov_inv_cg(
        self,
        X: torch.Tensor,
        b: torch.Tensor,
        damping: float,
        max_iter: int,
        tol: float,
        eps: float,
    ) -> torch.Tensor:
        if bool(self.cfg.cov_center):
            X = X - X.mean(dim=0, keepdim=True)

        T = int(X.shape[0])
        denom = float(max(T - 1 if bool(self.cfg.unbiased) else T, 1))

        def A(p: torch.Tensor) -> torch.Tensor:
            return (X.T @ (X @ p)) / denom + damping * p

        x = torch.zeros_like(b)
        r = b - A(x)
        p = r.clone()
        rs_old = torch.dot(r, r)

        if float(rs_old) < tol * tol:
            return x

        for _ in range(max_iter):
            Ap = A(p)
            alpha = rs_old / (torch.dot(p, Ap) + eps)
            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = torch.dot(r, r)
            if float(rs_new) < tol * tol:
                break

            beta = rs_new / (rs_old + eps)
            p = r + beta * p
            rs_old = rs_new

        return x


# =========================================================
# Main editor
# =========================================================

class CommonGpopEditor:
    """
    Flow:
      1) get common dims
      2) state.build_current(...)
      3) cold start: update EMA only
      4) else build v_ref
      5) edit only negative tasks on common dims
      6) update EMA using current valid rows
    """

    def __init__(
        self,
        named_params: List[Tuple[str, torch.nn.Parameter]],
        common_param_filter: Callable[[str], bool],
        cfg: Optional[CommonGpopConfig] = None,
    ):
        self.cfg = cfg or CommonGpopConfig()
        self.cfg.validate()

        self._common_ids_cpu = build_common_ids(named_params, common_param_filter)
        self.state = CommonGpopState(
            ema_beta=float(self.cfg.ema_beta),
            gpop_keys=self.cfg.gpop_keys,
        )
        self.ref_builder = GpopRefBuilder(self.cfg)

    @torch.no_grad()
    def apply(
        self,
        G: torch.Tensor, # [T, P]
        loss_keys: List[str],   # [T]
        weights: Optional[torch.Tensor] = None,   # [Tb]
        gpop_weights: Optional[torch.Tensor] = None,   # [Tb]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not torch.is_tensor(G):
            raise TypeError("[gpop] G must be a torch.Tensor")
        if G.ndim != 2:
            raise ValueError(f"[gpop] G must have shape [T,P], got {tuple(G.shape)}")
        if G.shape[0] != len(loss_keys):
            raise ValueError(f"[gpop] G.shape[0] ({G.shape[0]}) != len(loss_keys) ({len(loss_keys)})")
        if weights is not None and weights.shape[0] != len(self.cfg.gpop_keys):
            raise ValueError(f"[gpop] weights.shape[0] ({weights.shape[0]}) != len(gpop_keys) ({len(self.cfg.gpop_keys)})")
        if gpop_weights is not None and gpop_weights.shape[0] != len(self.cfg.gpop_keys):
            raise ValueError(f"[gpop] gpop_weights.shape[0] ({gpop_weights.shape[0]}) != len(gpop_keys) ({len(self.cfg.gpop_keys)})")

        device, dtype = G.device, G.dtype
        if weights is not None:
            weights = weights.to(device=device, dtype=dtype)

        if gpop_weights is not None:
            gpop_weights = gpop_weights.to(device=device, dtype=dtype)
        col_ids = self._common_ids_cpu.to(device=device)

        base_stats = {
            "enabled": 1.0,
            "num_tasks": float(G.shape[0]),
            "num_params": float(G.shape[1]),
            "num_common": float(col_ids.numel()),
        }

        if col_ids.numel() == 0:
            raise RuntimeError("[gpop] no common dimensions but enabled gpop surgery")

        G_common = G.index_select(1, col_ids)   # [T, Pc]
        
        # current batch -> state cache
        self.state.build_current(
            G_common=G_common,
            loss_keys=loss_keys,
            weights=weights,
        )
        
        G_cur = self.state.cur_G
        ref_G = self.state.ref_G

        # no matched keys under schema
        if G_cur is None or G_cur.shape[0] == 0:
            stats = _as_tensor_dict(base_stats, device, dtype)
            stats["warmup"] = torch.tensor(0.0, device=device, dtype=dtype)
            stats["matched_tasks"] = torch.tensor(0.0, device=device, dtype=dtype)
            return G, _prefix(stats, "gpop")

        # cold start: first valid update only initializes EMA
        if ref_G is None:
            self.state.update_ema()
            stats = _as_tensor_dict(base_stats, device, dtype)
            stats["warmup"] = torch.tensor(1.0, device=device, dtype=dtype)
            stats["matched_tasks"] = torch.tensor(float(G_cur.shape[0]), device=device, dtype=dtype)
            return G, _prefix(stats, "gpop")

        # build reference from EMA base
        g_ref = self.ref_builder.build_g_ref(
            G_cur=G_cur,
            G_ref=ref_G,
            activated_rows=self.state.activated_rows,
            gpop_weights=gpop_weights,
        )

        # edit all current tasks on common dims
        G_cur_new, edit_stats = self._edit_common(G_cur, g_ref)
        
        # update G in place
        G[self.state.cur_src_ids[:, None], col_ids[None, :]] = G_cur_new   # [T, Pc]
        
        # update EMA after edit decision; state cache keeps current matched rows
        self.state.update_ema()

        stats = _as_tensor_dict(base_stats, device, dtype)
        stats["warmup"] = torch.tensor(0.0, device=device, dtype=dtype)
        stats["matched_tasks"] = torch.tensor(float(G_cur.shape[0]), device=device, dtype=dtype)
        stats.update(_prefix(edit_stats, "edit"))
        return G, _prefix(stats, "gpop")

    @torch.no_grad()
    def _edit_common(self, G_cur: torch.Tensor, g_ref: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.cfg.edit_kind == "project":
            return self._project_negative_tasks(G_cur, g_ref)
        raise ValueError(f"[gpop] unsupported edit_kind: {self.cfg.edit_kind}")

    @torch.no_grad()
    def _project_negative_tasks(
        self,
        G_cur: torch.Tensor,   # [T, Pc]
        g_ref: torch.Tensor,      # [Pc]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eps = float(self.cfg.eps)
        g_ref_norm_sq = torch.dot(g_ref, g_ref).clamp_min(eps)

        dots_before = G_cur @ g_ref
        neg_mask = dots_before < 0
        coeff = torch.zeros_like(dots_before)
        coeff[neg_mask] = dots_before[neg_mask] / g_ref_norm_sq

        G_new = G_cur.clone()
        if neg_mask.any():
            G_new[neg_mask] = G_new[neg_mask] - coeff[neg_mask].unsqueeze(1) * g_ref.unsqueeze(0)

        dots_after = G_new @ g_ref

        stats = {
            "num_negative": neg_mask.float().sum(),
            "changed_ratio": neg_mask.float().mean(),
            "dot_before.mean": dots_before.mean(),
            "dot_after.mean": dots_after.mean(),
            "g_ref.norm": _safe_norm(g_ref, eps),
        }
        return G_new, _as_tensor_dict(stats, G_cur.device, G_cur.dtype)

    def state_dict(self) -> Dict:
        return {
            "cfg": dict(self.cfg.__dict__),
            "state": self.state.state_dict(),
        }

    def load_state_dict(self, sd: Dict):
        if "state" in sd:
            self.state.load_state_dict(sd["state"])