from __future__ import annotations

import logging

import torch

from o3b.model.model import OD3D_Model, register_model

logger = logging.getLogger(__name__)


@register_model("SigLIP2")
class SigLIP2Model(OD3D_Model):
    """Self-rendering per-vertex feature extractor using SigLIP2 as the dense 2D encoder.

    Reuses the Diff3F rendering/projection/aggregation pipeline
    (o3b.model.diff3f.diff3f.get_features_per_vertex) via the extractor_fn hook —
    only the 2D feature extraction (SigLIP2Extractor) differs from Diff3F.
    """

    def __init__(
        self,
        hub_model: str = "google/siglip2-base-patch16-384",
        num_views: int = 100,
        resolution: int = 512,
        tolerance: float = 0.004,
        freeze: bool = True,
        aggregation_mode: str = "mean",
    ):
        super().__init__()
        self.hub_model = hub_model
        self.num_views = num_views
        self.resolution = resolution
        self.tolerance = tolerance
        self.freeze = freeze
        self.aggregation_mode = aggregation_mode
        self.out_dim = None  # known only once the extractor loads the HF config

    def _forward_object_batch(self, batch):
        import gc
        from o3b.model.diff3f.diff3f_utils import compute_features
        from o3b.model.diff3f.diff3f.dataloaders.mesh_container import MeshContainer
        from o3b.model.siglip2.extractor import SigLIP2Extractor

        mesh = batch.mesh
        if mesh is None:
            return batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extractor = SigLIP2Extractor(self.hub_model, device=device)
        self.out_dim = extractor.feature_dims

        source_mesh = MeshContainer(vert=mesh.verts, face=mesh.faces)
        feats = compute_features(
            device, None, None, source_mesh,
            None, self.num_views, self.resolution, self.resolution,
            self.tolerance, 1, False,
            extractor_fn=extractor,
            aggregation_mode=self.aggregation_mode,
        )  # (V, C) float cpu, or PerVertexFeatures if aggregation_mode="all_views"

        del extractor
        torch.cuda.empty_cache()
        gc.collect()

        B = batch.verts3d.shape[0] if batch.verts3d is not None else 1

        if self.aggregation_mode == "all_views":
            # Mirrors Diff3FModel._forward_object_batch's all_views branch: the mean
            # representation still lands on verts3d_feats unchanged, all_views is a
            # parallel, opt-in field.
            mean_feats      = feats.mean.float()
            all_views_feats = feats.all_views
            all_views_mask  = feats.all_views_mask
            V, K = all_views_feats.shape[0], all_views_feats.shape[1]
            batch.verts3d_feats = mean_feats.unsqueeze(0).expand(B, V, -1).contiguous()
            batch.verts3d_feats_all_views = all_views_feats.unsqueeze(0).expand(B, V, K, -1).contiguous()
            batch.verts3d_feats_all_views_mask = all_views_mask.unsqueeze(0).expand(B, V, K).contiguous()
            return batch

        feats = feats.float()
        V = feats.shape[0]
        batch.verts3d_feats = feats.unsqueeze(0).expand(B, V, -1).contiguous()
        return batch

    def forward(self, frames_gt, frames_pred=None):
        from o3b.data.datatypes.object import ObjectBatch
        if isinstance(frames_gt, ObjectBatch):
            return self._forward_object_batch(frames_gt)
        raise NotImplementedError(
            "SigLIP2Model only supports self-rendering ObjectBatch inputs; "
            "frame-based (pre-rendered) usage is not implemented."
        )
