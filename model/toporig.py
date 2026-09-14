from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _faces_to_edges(
    faces: torch.Tensor,
    num_vertices: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    faces = faces.to(device=device, dtype=torch.long)
    if faces.numel() == 0:
        idx = torch.arange(num_vertices, device=device)
        return idx, idx

    valid = (faces >= 0).all(dim=-1)
    faces = faces[valid]
    if faces.numel() == 0:
        idx = torch.arange(num_vertices, device=device)
        return idx, idx

    i, j, k = faces.unbind(dim=-1)
    src = torch.cat([i, j, j, k, k, i], dim=0)
    dst = torch.cat([j, i, k, j, i, k], dim=0)

    in_range = (
        (src >= 0)
        & (src < num_vertices)
        & (dst >= 0)
        & (dst < num_vertices)
    )
    src = src[in_range]
    dst = dst[in_range]

    self_edges = torch.arange(num_vertices, device=device)
    src = torch.cat([src, self_edges], dim=0)
    dst = torch.cat([dst, self_edges], dim=0)
    return src, dst


def _edge_mean(
    x: torch.Tensor,
    edge_index: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    src, dst = edge_index
    batch_size, num_vertices, channels = x.shape

    out = x.new_zeros(batch_size, num_vertices, channels)
    out.index_add_(1, dst, x[:, src, :])

    degree = x.new_zeros(num_vertices)
    degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    return out / degree.clamp_min(1.0).view(1, num_vertices, 1)


def _dense_adj_mean(x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    if adjacency.dim() == 2:
        adjacency = adjacency.unsqueeze(0).expand(x.shape[0], -1, -1)

    adjacency = adjacency.to(device=x.device, dtype=x.dtype)
    degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return torch.bmm(adjacency / degree, x)


def _neighbor_mean(
    x: torch.Tensor,
    faces: Optional[torch.Tensor] = None,
    adjacency: Optional[torch.Tensor] = None,
    edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if adjacency is not None:
        return _dense_adj_mean(x, adjacency)

    if edge_index is not None:
        return _edge_mean(x, edge_index)

    if faces is None:
        return x

    if faces.dim() == 2:
        edge_index = _faces_to_edges(faces, x.shape[1], x.device)
        return _edge_mean(x, edge_index)

    if faces.dim() != 3:
        raise ValueError("faces must have shape [F, 3] or [B, F, 3].")

    if faces.shape[0] != x.shape[0]:
        raise ValueError("batched faces must have the same batch size as x.")

    outputs = []
    for batch_idx in range(x.shape[0]):
        batch_edges = _faces_to_edges(faces[batch_idx], x.shape[1], x.device)
        outputs.append(_edge_mean(x[batch_idx : batch_idx + 1], batch_edges))
    return torch.cat(outputs, dim=0)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        if num_layers == 1:
            dims = [in_dim, out_dim]
        else:
            dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        layers = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurfaceDiffusionBlock(nn.Module):
    """A lightweight DiffusionNet-like block for per-vertex mesh features."""

    def __init__(self, feature_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_mlp = MLP(
            in_dim=feature_dim * 3,
            hidden_dim=feature_dim,
            out_dim=feature_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        faces: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        diffused = _neighbor_mean(
            x,
            faces=faces,
            adjacency=adjacency,
            edge_index=edge_index,
        )
        gradient = diffused - x
        update = self.spatial_mlp(torch.cat([x, diffused, gradient], dim=-1))
        return self.norm(x + update)


class ConditionalDiffusionBlock(nn.Module):
    """Diffusion block with FACS/global conditioning injected per vertex."""

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial_mlp = MLP(
            in_dim=feature_dim * 3,
            hidden_dim=feature_dim,
            out_dim=feature_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.condition_mlp = MLP(
            in_dim=feature_dim + condition_dim,
            hidden_dim=feature_dim,
            out_dim=feature_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        faces: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        diffused = _neighbor_mean(
            x,
            faces=faces,
            adjacency=adjacency,
            edge_index=edge_index,
        )
        gradient = diffused - x
        spatial = self.spatial_mlp(torch.cat([x, diffused, gradient], dim=-1))

        condition = condition.unsqueeze(1).expand(-1, x.shape[1], -1)
        update = self.condition_mlp(torch.cat([spatial, condition], dim=-1))
        return self.norm(x + update)


class GlobalEncoder(nn.Module):
    """Two-layer mesh encoder with global average pooling, as in Fig. 3."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        global_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_mlp = MLP(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [SurfaceDiffusionBlock(hidden_dim, dropout=dropout) for _ in range(2)]
        )
        self.output_mlp = MLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=global_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        vertex_features: torch.Tensor,
        faces: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.input_mlp(vertex_features)
        for block in self.blocks:
            x = block(
                x,
                faces=faces,
                adjacency=adjacency,
                edge_index=edge_index,
            )
        return self.output_mlp(x).mean(dim=1)


class TopoRig(nn.Module):
    """
    Mesh deformation network inspired by the RigAnyFace/RAF Fig. 3 architecture.

    Args:
        facs_dim: Number of FACS/action-unit controls in the pose vector.
        hidden_dim: Width of the main conditional diffusion stack.
        num_blocks: Number of conditional diffusion blocks in the main network.
        use_global_encoder: If True, encode the neutral mesh with a 2-layer
            global encoder and concatenate that embedding with the FACS vector.
        global_dim: Size of the optional global mesh embedding.
        condition_dim: Projected latent size injected into each diffusion block.
        input_dim: Per-vertex input feature size before positional encoding. The
            default expects position and normal concatenated as
            [x, y, z, nx, ny, nz]. Larger values reserve the remaining channels
            for optional auxiliary inputs such as landmark-relative features.
        position_encoding: Optional topology-agnostic positional encoding config.
            Set ``{"type": "fourier", "num_frequencies": 8}`` to concatenate
            Fourier features computed from per-mesh normalized coordinates.
        use_conditioned_output_head: If True, concatenate condition features into
            the final output MLP. Set False only for loading legacy checkpoints
            trained before the output head received condition features.
        output_scale: Multiplier applied to predicted displacements.
        output_init: Initialization for the final displacement layer. ``"zero"``
            starts new models at neutral displacement, which is usually the
            right prior for small blendshape deltas.
        output_init_std: Standard deviation used when ``output_init="normal"``.
    """

    def __init__(
        self,
        facs_dim: int = 96,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        use_global_encoder: bool = False,
        global_dim: int = 128,
        condition_dim: int = 128,
        input_dim: int = 6,
        dropout: float = 0.0,
        output_scale: float = 1.0,
        output_init: str = "zero",
        output_init_std: float = 1.0e-5,
        position_encoding: Optional[Mapping[str, Any] | str] = None,
        use_conditioned_output_head: bool = True,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        if input_dim < 6:
            raise ValueError(
                "TopoRig requires input_dim >= 6 for base vertex features "
                "[x, y, z, nx, ny, nz]."
            )
        self.input_dim = input_dim
        self.auxiliary_input_dim = input_dim - 6
        self.facs_dim = facs_dim
        self.use_global_encoder = use_global_encoder
        self.use_conditioned_output_head = bool(use_conditioned_output_head)
        self.output_scale = float(output_scale)
        (
            self.position_encoding_type,
            self.position_encoding_num_frequencies,
            self.position_encoding_include_raw,
            self.normalize_base_positions,
        ) = self._parse_position_encoding(position_encoding)
        self.position_encoding_frequency_scale = math.pi

        fourier_dim = 0
        if self.position_encoding_type == "fourier":
            fourier_dim = 3 * 2 * self.position_encoding_num_frequencies
            if self.position_encoding_include_raw:
                fourier_dim += 3
            frequencies = torch.pow(
                2.0,
                torch.arange(self.position_encoding_num_frequencies).float(),
            )
        else:
            frequencies = torch.empty(0)
        self.register_buffer(
            "position_encoding_frequencies",
            frequencies,
            persistent=False,
        )
        model_input_dim = input_dim + fourier_dim

        self.vertex_mlp = MLP(
            in_dim=model_input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
            dropout=dropout,
        )

        self.global_encoder: Optional[GlobalEncoder]
        if use_global_encoder:
            self.global_encoder = GlobalEncoder(
                input_dim=model_input_dim,
                hidden_dim=max(hidden_dim // 2, condition_dim),
                global_dim=global_dim,
                dropout=dropout,
            )
            latent_dim = facs_dim + global_dim
        else:
            self.global_encoder = None
            latent_dim = facs_dim

        self.condition_mlp = MLP(
            in_dim=latent_dim,
            hidden_dim=condition_dim,
            out_dim=condition_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.blocks = nn.ModuleList(
            [
                ConditionalDiffusionBlock(
                    feature_dim=hidden_dim,
                    condition_dim=condition_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        output_input_dim = hidden_dim
        if self.use_conditioned_output_head:
            output_input_dim += condition_dim
        self.output_mlp = MLP(
            in_dim=output_input_dim,
            hidden_dim=hidden_dim,
            out_dim=3,
            num_layers=3,
            dropout=dropout,
        )
        self._initialize_output_layer(output_init, output_init_std)

    def forward(
        self,
        vertices: torch.Tensor,
        facs: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        faces: Optional[torch.Tensor] = None,
        adjacency: Optional[torch.Tensor] = None,
        landmark_features: Optional[torch.Tensor] = None,
        return_deformed: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict per-vertex 3D displacement.

        Args:
            vertices: Neutral mesh vertices with shape [B, V, 3].
            facs: FACS pose vector with shape [B, C] or [B, P, C].
            normals: Optional per-vertex normals with shape [B, V, 3].
            faces: Optional triangle indices with shape [F, 3] or [B, F, 3].
            adjacency: Optional dense adjacency with shape [V, V] or [B, V, V].
            landmark_features: Optional auxiliary per-vertex features with shape
                [B, V, input_dim - 6].
            return_deformed: If True, return (deformed_vertices, displacement).

        Returns:
            Displacement with shape [B, V, 3] for one pose or [B, P, V, 3]
            for multiple poses. If return_deformed=True, the first item is
            vertices + displacement with the same output shape.
        """
        if vertices.dim() != 3 or vertices.shape[-1] != 3:
            raise ValueError("vertices must have shape [B, V, 3].")

        if normals is None:
            normals = torch.zeros_like(vertices)
        elif normals.shape != vertices.shape:
            raise ValueError("normals must have the same shape as vertices.")
        else:
            normals = normals.to(device=vertices.device, dtype=vertices.dtype)
            normals = F.normalize(normals, dim=-1, eps=1e-6)

        if facs.dim() == 2:
            displacement = self._forward_single_pose(
                vertices=vertices,
                normals=normals,
                facs=facs,
                faces=faces,
                adjacency=adjacency,
                landmark_features=landmark_features,
            )
            if return_deformed:
                return vertices + displacement, displacement
            return displacement

        if facs.dim() != 3:
            raise ValueError("facs must have shape [B, C] or [B, P, C].")

        batch_size, num_poses, facs_dim = facs.shape
        if batch_size != vertices.shape[0]:
            raise ValueError("facs and vertices must have the same batch size.")

        vertices_rep = (
            vertices.unsqueeze(1)
            .expand(batch_size, num_poses, -1, -1)
            .reshape(batch_size * num_poses, vertices.shape[1], 3)
        )
        normals_rep = (
            normals.unsqueeze(1)
            .expand(batch_size, num_poses, -1, -1)
            .reshape(batch_size * num_poses, normals.shape[1], 3)
        )
        landmark_features_rep = None
        if landmark_features is not None:
            if landmark_features.dim() != 3 or landmark_features.shape[:2] != vertices.shape[:2]:
                raise ValueError(
                    "landmark_features must have shape [B, V, input_dim - 6]."
                )
            landmark_features_rep = (
                landmark_features.unsqueeze(1)
                .expand(batch_size, num_poses, -1, -1)
                .reshape(
                    batch_size * num_poses,
                    landmark_features.shape[1],
                    landmark_features.shape[2],
                )
            )
        facs_rep = facs.reshape(batch_size * num_poses, facs_dim)

        faces_rep = faces
        if faces is not None and faces.dim() == 3:
            faces_rep = (
                faces.unsqueeze(1)
                .expand(batch_size, num_poses, -1, -1)
                .reshape(batch_size * num_poses, faces.shape[1], 3)
            )

        adjacency_rep = adjacency
        if adjacency is not None and adjacency.dim() == 3:
            adjacency_rep = (
                adjacency.unsqueeze(1)
                .expand(batch_size, num_poses, -1, -1)
                .reshape(batch_size * num_poses, adjacency.shape[1], adjacency.shape[2])
            )

        displacement = self._forward_single_pose(
            vertices=vertices_rep,
            normals=normals_rep,
            facs=facs_rep,
            faces=faces_rep,
            adjacency=adjacency_rep,
            landmark_features=landmark_features_rep,
        ).reshape(batch_size, num_poses, vertices.shape[1], 3)

        if return_deformed:
            deformed = vertices.unsqueeze(1) + displacement
            return deformed, displacement
        return displacement

    def _forward_single_pose(
        self,
        vertices: torch.Tensor,
        normals: torch.Tensor,
        facs: torch.Tensor,
        faces: Optional[torch.Tensor],
        adjacency: Optional[torch.Tensor],
        landmark_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if facs.dim() != 2 or facs.shape[-1] != self.facs_dim:
            raise ValueError(f"facs must have shape [B, {self.facs_dim}].")
        if facs.shape[0] != vertices.shape[0]:
            raise ValueError("facs and vertices must have the same batch size.")

        faces = faces.to(vertices.device) if faces is not None else None
        adjacency = adjacency.to(vertices.device) if adjacency is not None else None
        edge_index = None
        if adjacency is None and faces is not None and faces.dim() == 2:
            edge_index = _faces_to_edges(faces, vertices.shape[1], vertices.device)

        vertex_features = self._vertex_features(
            vertices,
            normals,
            landmark_features=landmark_features,
        )
        x = self.vertex_mlp(vertex_features)

        latent = facs.to(device=vertices.device, dtype=vertices.dtype)
        if self.global_encoder is not None:
            global_embedding = self.global_encoder(
                vertex_features,
                faces=faces,
                adjacency=adjacency,
                edge_index=edge_index,
            )
            latent = torch.cat([latent, global_embedding], dim=-1)
        condition = self.condition_mlp(latent)

        for block in self.blocks:
            x = block(
                x,
                condition=condition,
                faces=faces,
                adjacency=adjacency,
                edge_index=edge_index,
            )
        output_features = x
        if self.use_conditioned_output_head:
            condition_features = condition.unsqueeze(1).expand(-1, x.shape[1], -1)
            output_features = torch.cat((x, condition_features), dim=-1)
        return self.output_mlp(output_features) * self.output_scale

    @staticmethod
    def _parse_position_encoding(
        config: Optional[Mapping[str, Any] | str],
    ) -> tuple[str, int, bool, bool]:
        if config is None:
            return "none", 0, False, False
        if isinstance(config, str):
            config = {"type": config}
        if not isinstance(config, Mapping):
            raise ValueError("position_encoding must be a mapping, string, or null.")

        encoding_type = str(config.get("type", "none")).lower()
        if encoding_type in {"none", "off", "disabled", "false"}:
            return "none", 0, False, bool(config.get("normalize_positions", False))
        if encoding_type not in {"fourier", "sinusoidal"}:
            raise ValueError(
                "position_encoding.type must be one of: none, fourier, sinusoidal."
            )

        num_frequencies = int(config.get("num_frequencies", 8))
        if num_frequencies < 1:
            raise ValueError("position_encoding.num_frequencies must be at least 1.")
        include_raw = bool(config.get("include_raw", False))
        normalize_positions = bool(config.get("normalize_positions", True))
        return "fourier", num_frequencies, include_raw, normalize_positions

    def _vertex_features(
        self,
        vertices: torch.Tensor,
        normals: torch.Tensor,
        landmark_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized_vertices = self._normalized_vertices(vertices)
        positions = normalized_vertices if self.normalize_base_positions else vertices
        features = [positions, normals]
        if self.auxiliary_input_dim > 0:
            if landmark_features is None:
                landmark_features = vertices.new_zeros(
                    vertices.shape[0],
                    vertices.shape[1],
                    self.auxiliary_input_dim,
                )
            elif landmark_features.shape != (
                vertices.shape[0],
                vertices.shape[1],
                self.auxiliary_input_dim,
            ):
                raise ValueError(
                    "landmark_features must have shape "
                    f"[B, V, {self.auxiliary_input_dim}]."
                )
            else:
                landmark_features = landmark_features.to(
                    device=vertices.device,
                    dtype=vertices.dtype,
                )
            features.append(landmark_features)
        elif landmark_features is not None and landmark_features.numel() > 0:
            raise ValueError(
                "landmark_features were provided, but input_dim does not reserve "
                "auxiliary channels."
            )
        if self.position_encoding_type == "fourier":
            features.append(self._fourier_position_features(normalized_vertices))
        return torch.cat(features, dim=-1)

    @staticmethod
    def _normalized_vertices(vertices: torch.Tensor) -> torch.Tensor:
        center = vertices.mean(dim=1, keepdim=True)
        centered = vertices - center
        scale = torch.linalg.vector_norm(centered, dim=-1).amax(
            dim=1,
            keepdim=True,
        )
        return centered / scale.clamp_min(1.0e-6).unsqueeze(-1)

    def _fourier_position_features(
        self,
        normalized_vertices: torch.Tensor,
    ) -> torch.Tensor:
        frequencies = self.position_encoding_frequencies.to(
            device=normalized_vertices.device,
            dtype=normalized_vertices.dtype,
        )
        angles = (
            normalized_vertices.unsqueeze(-2)
            * frequencies.view(1, 1, -1, 1)
            * self.position_encoding_frequency_scale
        )
        encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-2)
        encoded = encoded.flatten(start_dim=-2)
        if self.position_encoding_include_raw:
            encoded = torch.cat((normalized_vertices, encoded), dim=-1)
        return encoded

    def _initialize_output_layer(
        self,
        output_init: str,
        output_init_std: float,
    ) -> None:
        output_init = output_init.lower()
        if output_init == "default":
            return

        final_linear = None
        for module in self.output_mlp.net.modules():
            if isinstance(module, nn.Linear):
                final_linear = module
        if final_linear is None:
            raise RuntimeError("TopoRig output MLP does not contain a Linear layer.")

        if output_init == "zero":
            nn.init.zeros_(final_linear.weight)
        elif output_init == "normal":
            nn.init.normal_(final_linear.weight, mean=0.0, std=float(output_init_std))
        else:
            raise ValueError(
                "output_init must be one of: 'zero', 'normal', or 'default'."
            )
        nn.init.zeros_(final_linear.bias)


def _test() -> None:
    torch.manual_seed(7)

    batch_size = 2
    num_vertices = 128
    facs_dim = 96

    vertices = torch.randn(batch_size, num_vertices, 3)
    normals = F.normalize(torch.randn(batch_size, num_vertices, 3), dim=-1)
    facs = torch.randn(batch_size, facs_dim)

    face_ids = torch.arange(num_vertices - 2)
    faces = torch.stack([face_ids, face_ids + 1, face_ids + 2], dim=-1)

    model = TopoRig(
        facs_dim=facs_dim,
        hidden_dim=64,
        num_blocks=2,
        use_global_encoder=False,
    )
    output = model(vertices, facs, normals=normals, faces=faces)

    model_with_global = TopoRig(
        facs_dim=facs_dim,
        hidden_dim=64,
        num_blocks=2,
        use_global_encoder=True,
    )
    output_with_global = model_with_global(
        vertices,
        facs,
        normals=normals,
        faces=faces,
    )

    print(f"vertices input shape: {tuple(vertices.shape)}")
    print(f"normals input shape:  {tuple(normals.shape)}")
    print(f"facs input shape:     {tuple(facs.shape)}")
    print(f"faces input shape:    {tuple(faces.shape)}")
    print(f"output shape:         {tuple(output.shape)}")
    print(f"global output shape:  {tuple(output_with_global.shape)}")


if __name__ == "__main__":
    _test()
