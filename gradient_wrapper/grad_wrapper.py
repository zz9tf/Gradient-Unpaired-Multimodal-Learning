import torch
from gradient_wrapper.grad_gpop import CommonGpopEditor
from gradient_wrapper.grad_block_monitor import GradientBlockState
from typing import Dict, List, Optional, Callable, Tuple, Any

def default_monitor_block_fn(name: str) -> str:
    """
    Build a coarse block id from a parameter name.

    Args:
        name: Full parameter name from ``named_parameters``.

    Returns:
        First two name segments joined by dot. Falls back to first segment.
    """
    n = name[7:] if name.startswith("module.") else name
    base = n.replace(".weight", "").replace(".bias", "")
    parts = base.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")


def build_common_slices(
    named_params,
    common_param_filter: Callable[[str], bool],
) -> List[Tuple[int, int]]:
    slices = []
    offset = 0
    for name, p in named_params:
        n_elem = int(p.numel())
        if bool(common_param_filter(name)):
            slices.append((offset, offset + n_elem))
        offset += n_elem
    return slices

# ----------------------------
# basic helpers
# ----------------------------
def named_params(
    model: torch.nn.Module,
    only: Optional[Callable[[str], bool]] = None
) -> List[Tuple[str, torch.nn.Parameter]]:
    out = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if only is None or bool(only(n)):
            out.append((n, p))
    return out


def safe_grads(grads, params):
    return [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params)]


def flatten(grads: List[torch.Tensor]) -> torch.Tensor:
    if len(grads) == 0:
        return torch.zeros(0)
    return torch.cat([g.reshape(-1) for g in grads], dim=0)


def unflatten(vec: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out, off = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[off: off + n].view_as(p))
        off += n
    return out


def gradvec(
    loss: torch.Tensor,
    params: List[torch.nn.Parameter],
    retain_graph: bool,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    grads = safe_grads(grads, params)
    return flatten(grads)


# ----------------------------
# wrapper
# ----------------------------
class GradWrapper:
    """
    Runtime only.
    Responsibilities:
      1) compute per-loss grads G=[T,P]
      2) call pre-monitor
      3) call gpop operator
      4) call post-monitor
      5) merge final gradient and write back .grad

    Not responsible for:
      - creating monitor
      - creating gpop
      - old aggregation mode zoo
    """

    def __init__(
        self,
        model: torch.nn.Module,
        param_filter: Optional[Callable[[str, torch.nn.Parameter], bool]] = None,
        monitor_pre: Optional[GradientBlockState] = None,
        monitor_post: Optional[GradientBlockState] = None,
        gpop: Optional[CommonGpopEditor] = None,
        verbose: bool = True,
    ):
        self.model = model

        named_all = named_params(model, only=param_filter)
        if len(named_all) == 0:
            raise ValueError("No trainable params selected.")

        self._named_all = named_all
        self.all_names = [n for n, _ in named_all]
        self.params = [p for _, p in named_all]

        # injected components
        self.monitor_pre = monitor_pre
        self.monitor_post = monitor_post
        self.gpop = gpop

        # bookkeeping
        self._step: int = 0
        self.last_stats: Dict[str, torch.Tensor] = {}

        if verbose:
            print("[GradWrapper] tensors:", len(self.params), "examples:", self.all_names[:5])
            print("[GradWrapper] monitor_pre:", self.monitor_pre.name if self.monitor_pre is not None else None)
            print("[GradWrapper] monitor_post:", self.monitor_post.name if self.monitor_post is not None else None)
            print("[GradWrapper] gpop:", self.gpop.cfg.gpop_keys if self.gpop is not None else None)

    # ----------------------------
    # helpers
    # ----------------------------
    def _writeback(self, g_final: torch.Tensor):
        grads = unflatten(g_final, self.params)
        for p, g in zip(self.params, grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

    # ----------------------------
    # main
    # ----------------------------
    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        task_weights: Optional[torch.Tensor] = None,
        gpop_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            losses: dict loss_key -> scalar loss tensor
            task_weights: optional 1D tensor of length T = len(losses), same order as
                ``list(losses.keys())``, applied as row weights before summing into the
                final gradient. If None, each task row is weighted by 1.
            gpop_weights: optional 1D tensor aligned to ``CommonGpopEditor.cfg.gpop_keys``
                (full schema length), passed to gpop when enabled.
        Returns:
            stats dict
        """
        if len(losses) == 0:
            raise ValueError("losses is empty.")

        self._step += 1
        loss_keys = list(losses.keys())
        T = len(loss_keys)
        # 1) compute per-loss gradients
        g_list = []
        for i, k in enumerate(loss_keys):
            retain_graph = (i != T - 1)
            g = gradvec(losses[k], self.params, retain_graph=retain_graph)
            g_list.append(g)
        G = torch.stack(g_list, dim=0)  # [T, P]

        stats: Dict[str, torch.Tensor] = {}
        stats["wrapper.step"] = G.new_tensor(float(self._step))
        stats["wrapper.num_losses"] = G.new_tensor(float(T))

        # 2) pre-monitor
        if self.monitor_pre is not None:
            st_pre = self.monitor_pre.monitor(
                G.detach(),
                loss_keys=loss_keys,
                step=self._step,
            )
            stats.update(st_pre)

        # 3) gpop apply
        G_used = G
        if self.gpop is not None:
            G_used, st_gpop = self.gpop.apply(
                G=G,
                loss_keys=loss_keys,
                weights=task_weights,
                gpop_weights=gpop_weights,
            )
            
            stats.update(st_gpop)

        # 4) post-monitor
        if self.monitor_post is not None and self.gpop is not None:
            st_post = self.monitor_post.monitor(
                G_used.detach(),
                loss_keys=loss_keys,
                step=self._step,
            )
            stats.update(st_post)

        # 5) merge final gradient and write back
        g_final = G_used.sum(dim=0) # [P]
        self._writeback(g_final)

        stats = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in stats.items()}

        self.last_stats = stats
        return stats

    # ----------------------------
    # state I/O
    # ----------------------------
    def state_dict(self) -> Dict[str, Any]:
        out = {
            "step": self._step,
        }
        if self.monitor_pre is not None and hasattr(self.monitor_pre, "state_dict"):
            out["monitor_pre"] = self.monitor_pre.state_dict()
        if self.monitor_post is not None and hasattr(self.monitor_post, "state_dict"):
            out["monitor_post"] = self.monitor_post.state_dict()
        if self.gpop is not None and hasattr(self.gpop, "state_dict"):
            out["gpop"] = self.gpop.state_dict()
        return out

    def load_state_dict(self, sd: Dict[str, Any]):
        self._step = int(sd.get("step", 0))

        if self.monitor_pre is not None and "monitor_pre" in sd and hasattr(self.monitor_pre, "load_state_dict"):
            self.monitor_pre.load_state_dict(sd["monitor_pre"])

        if self.monitor_post is not None and "monitor_post" in sd and hasattr(self.monitor_post, "load_state_dict"):
            self.monitor_post.load_state_dict(sd["monitor_post"])

        if self.gpop is not None and "gpop" in sd and hasattr(self.gpop, "load_state_dict"):
            self.gpop.load_state_dict(sd["gpop"])