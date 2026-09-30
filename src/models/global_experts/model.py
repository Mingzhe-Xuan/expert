"""GMTNet encoder, canonical Full-PG experts, hierarchical crystal residual fusion."""

from dataclasses import dataclass
import hashlib
from typing import Mapping
import torch
from torch import nn

from ...baselines.gmtnet import GMTNET_OFFICIAL_COMMIT
from ...baselines.gmtnet import configure_equivariant_attention
from ...experts.modules import FullPointGroupExpert
from ...symmetry import ParentDAGSpec, PointGroupAncestorDAG, PointGroupRegistry
from ...symmetry.registry import _layout_irreps
from e3nn import o3
from .config import GlobalExpertsConfig, layout_from_irreps
from .adapter import IdentityMessageAdapter
from .routing import HierarchicalChainRouter


@dataclass(frozen=True)
class CrystalRouting:
    sample_id: str
    dag: ParentDAGSpec
    residuals: Mapping
    input_to_standard: torch.Tensor


class GlobalExpertsModel(nn.Module):
    def __init__(
        self,
        global_model,
        equality_adjustment,
        *,
        expert_numbers,
        edge_ids,
        config=GlobalExpertsConfig(),
        class_dag=None,
        registry=None,
    ):
        super().__init__()
        self.config = config
        self.global_model = configure_equivariant_attention(
            global_model, use_equiv_attn=config.use_equiv_attn
        )
        self.equality_adjustment = equality_adjustment
        self.class_dag = class_dag or PointGroupAncestorDAG.from_path()
        registry = registry or PointGroupRegistry()
        self.registry_sha256 = hashlib.sha256(registry.path.read_bytes()).hexdigest()
        if self.registry_sha256 != self.class_dag.asset_sha256:
            raise ValueError("PG registry and DAG asset hashes differ")
        self.expert_numbers = tuple(sorted(set(expert_numbers)))
        if not self.expert_numbers:
            raise ValueError("at least one registered expert required")
        self.global_irreps = o3.Irreps(global_model.equi_update.nlayer_3.out_irreps)
        self.expert_layout = layout_from_irreps(config.expert_irreps)
        self.expert_irreps = _layout_irreps(self.expert_layout)
        self.input_map = o3.Linear(self.global_irreps, self.expert_irreps)
        self.output_map = o3.Linear(self.expert_irreps, self.global_irreps)
        self.adapter = IdentityMessageAdapter(
            self.expert_layout,
            backend=config.adapter_backend,
            mmax=config.adapter_mmax,
            edge_lmax=config.adapter_edge_lmax,
            cutoff=config.adapter_cutoff,
            radial_width=config.adapter_radial_width,
            initial_logit=config.adapter_initial_logit,
        )
        self.experts = nn.ModuleDict(
            {
                str(number): FullPointGroupExpert(
                    registry[number], self.expert_layout, bypass_c1=config.bypass_c1
                )
                for number in self.expert_numbers
            }
        )
        self.router = HierarchicalChainRouter(
            edge_ids,
            initial_sigma=config.initial_sigma,
            sigma_floor=config.sigma_floor,
            temperature=config.chain_temperature,
            class_dag=self.class_dag,
        )
        self.branch_logit = nn.Parameter(
            torch.tensor(float(config.branch_initial_logit))
        )
        if config.freeze_global:
            self.global_model.requires_grad_(False)
            self.global_model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_global:
            self.global_model.eval()
        return self

    def metadata(self):
        return {
            "config": self.config.metadata(),
            "official_commit": GMTNET_OFFICIAL_COMMIT,
            "global_irreps": str(self.global_irreps),
            "expert_layout": self.expert_layout.to_spec(),
            "expert_numbers": self.expert_numbers,
            "edge_ids": self.router.edge_ids,
            "dag_sha256": self.class_dag.asset_sha256,
            "registry_sha256": self.registry_sha256,
            "input_map": "e3nn.o3.Linear",
            "output_map": "e3nn.o3.Linear",
            "edge_scale_parameterization": "softplus+sigma_floor",
            "branch_scale_parameterization": "sigmoid",
            "frame_convention": "lattice-polar-then-spglib-o3-v1",
            "chain_convention": "near-current-stick-breaking-immediate-parent-softmax-v1",
            "residual_convention": "minimum-oriented-parent-minus-child-relative-edge-rms-v1",
        }

    def encode_nodes(self, data):
        model = self.global_model
        nodes = model.atom_embedding(data.x)
        edges = model.rbf(-0.75 / data.edge_attr.norm(dim=1))
        for layer in model.att_layers:
            nodes = layer(nodes, data.edge_index, edges)
        return model.equi_update(data, nodes, data.edge_index, edges)

    def forward(
        self, data, feat_mask, equality, routing=None, *, return_diagnostics=False
    ):
        from torch_geometric.utils import scatter

        if data.x.ndim != 2 or not len(data.x) or data.batch.shape != (len(data.x),):
            raise ValueError(
                "nonempty node features and aligned batch membership required"
            )
        if (
            data.edge_index.ndim != 2
            or data.edge_index.shape[0] != 2
            or data.edge_attr.shape != (data.edge_index.shape[1], 3)
        ):
            raise ValueError("invalid graph edge geometry")
        if not torch.isfinite(data.edge_attr).all() or bool(
            (data.edge_attr.norm(dim=-1) <= 0).any()
        ):
            raise ValueError(
                "GMTNet edge distances must be finite and strictly positive"
            )
        count = int(data.batch.max()) + 1
        if int(data.batch.min()) < 0 or torch.unique(data.batch).numel() != count:
            raise ValueError(
                "batch membership must be contiguous with no empty crystals"
            )
        if feat_mask.shape != (count, self.global_irreps.dim, self.global_irreps.dim):
            raise ValueError("feature masks do not match crystal/carrier shape")
        if not self.config.auxiliary_enabled and not return_diagnostics:
            return self.global_model(data, feat_mask, equality)
        nodes = self.encode_nodes(data)
        global_features = scatter(nodes, data.batch, dim=0, reduce="mean")
        batch_size = len(global_features)
        diagnostics = {"global_features": global_features}
        fused = global_features
        if self.config.auxiliary_enabled:
            if routing is None or len(routing) != batch_size:
                raise ValueError("one routing record per crystal required")
            if len({row.sample_id for row in routing}) != batch_size:
                raise ValueError("duplicate sample IDs in routing batch")
            if hasattr(data, "sample_id") and tuple(data.sample_id) != tuple(
                r.sample_id for r in routing
            ):
                raise ValueError("graph/routing sample order mismatch")
            receiver, sender = data.edge_index
            if not torch.equal(data.batch[receiver], data.batch[sender]):
                raise ValueError("cross-crystal edges are forbidden")
            weights, transforms = [], []
            for row in routing:
                if row.sample_id != row.dag.material_id:
                    raise ValueError("routing sample/DAG identity mismatch")
                # Frames are detached geometric metadata; older e3nn Wigner generators
                # allocate on CPU even for CUDA angles. Build D on CPU, then transfer it.
                frame = row.input_to_standard.detach().to(device="cpu", dtype=nodes.dtype)
                if frame.shape != (3, 3) or not torch.allclose(
                    frame @ frame.T,
                    torch.eye(3, dtype=nodes.dtype),
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise ValueError("standard frame must be orthogonal")
                transforms.append(self.expert_irreps.D_from_matrix(frame).to(nodes))
                weights.append(self.router(row.dag, row.residuals))
            adapted = self.adapter(
                self.input_map(nodes), data.edge_index, data.edge_attr
            )
            rotations = torch.stack(transforms)
            standard = torch.einsum("nij,nj->ni", rotations[data.batch], adapted)
            pooled = [dict() for _ in routing]
            active = sorted({pg for w in weights for pg in w.alpha})
            for pg in active:
                if str(pg) not in self.experts:
                    raise ValueError(f"unregistered active expert {pg}")
                crystals = [i for i, w in enumerate(weights) if pg in w.alpha]
                selected = torch.zeros(
                    batch_size, dtype=torch.bool, device=nodes.device
                )
                selected[crystals] = True
                mask = selected[data.batch]
                expert_output = self.experts[str(pg)](standard[mask])
                restored = torch.einsum(
                    "nji,nj->ni", rotations[data.batch[mask]], expert_output
                )
                feature = scatter(
                    restored,
                    data.batch[mask],
                    dim=0,
                    dim_size=batch_size,
                    reduce="mean",
                )
                projected = self.output_map(feature)
                for i in crystals:
                    pooled[i][pg] = projected[i]
            auxiliary = torch.stack(
                [
                    torch.stack(
                        [pooled[i][pg] * weight for pg, weight in w.alpha.items()]
                    ).sum(0)
                    for i, w in enumerate(weights)
                ]
            )
            fused = global_features + self.branch_logit.sigmoid() * auxiliary
            diagnostics.update(
                routing=weights,
                pooled_experts=pooled,
                auxiliary=auxiliary,
                branch_scale=self.branch_logit.sigmoid(),
            )
        diagnostics["fused_features"] = fused
        if self.global_model.mask:
            fused = torch.bmm(feat_mask, fused.unsqueeze(-1)).squeeze(-1)
        prediction = self.equality_adjustment(
            equality, self.global_model.output_block(fused)
        )
        return (prediction, diagnostics) if return_diagnostics else prediction
