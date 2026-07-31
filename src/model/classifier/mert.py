"""MERT transfer model for closed-set artist classification."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class MERTArtistClassifier(nn.Module):
    """Pool pretrained MERT representations and predict an artist class.

    The default learns a softmax-weighted combination of all MERT layers while
    keeping the 95M-parameter backbone frozen.  ``unfreeze_last_n_layers``
    enables lightweight downstream fine-tuning when the linear-probe stage has
    converged.
    """

    def __init__(
        self,
        model_name_or_path: str = "m-a-p/MERT-v1-95M",
        *,
        num_classes: int,
        backbone: nn.Module | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_n_layers: int = 0,
        use_weighted_layers: bool = True,
        selected_layer: int = -1,
        classifier_hidden_dim: int = 512,
        dropout: float = 0.2,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if classifier_hidden_dim < 0 or not 0 <= dropout < 1:
            raise ValueError("invalid classifier_hidden_dim/dropout")
        if unfreeze_last_n_layers < 0:
            raise ValueError("unfreeze_last_n_layers must be non-negative")
        if backbone is None:
            from transformers import AutoModel

            backbone = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self.use_weighted_layers = bool(use_weighted_layers)
        self.selected_layer = int(selected_layer)
        config = getattr(backbone, "config", None)
        hidden_size = int(getattr(config, "hidden_size", 0))
        num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
        if hidden_size <= 0 or num_hidden_layers <= 0:
            raise ValueError("MERT backbone config needs hidden_size/num_hidden_layers")
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        if self.use_weighted_layers:
            self.layer_logits = nn.Parameter(torch.zeros(num_hidden_layers + 1))
        else:
            self.register_parameter("layer_logits", None)

        if freeze_backbone:
            self.backbone.requires_grad_(False)
        if unfreeze_last_n_layers:
            layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
            if layers is None:
                raise ValueError("cannot locate MERT encoder.layers for partial unfreezing")
            for layer in layers[-min(unfreeze_last_n_layers, len(layers)) :]:
                layer.requires_grad_(True)
        self.backbone_fully_frozen = not any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        if gradient_checkpointing and not self.backbone_fully_frozen:
            enable = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()

        if classifier_hidden_dim == 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, num_classes),
            )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_fully_frozen:
            self.backbone.eval()
        return self

    def _feature_mask(self, hidden: Tensor, attention_mask: Tensor | None) -> Tensor | None:
        if attention_mask is None:
            return None
        builder = getattr(self.backbone, "_get_feature_vector_attention_mask", None)
        if callable(builder):
            return builder(hidden.shape[1], attention_mask).to(hidden.device)
        return torch.ones(
            hidden.shape[:2],
            device=hidden.device,
            dtype=torch.bool,
        )

    def _combine_layers(self, hidden_states: tuple[Tensor, ...]) -> Tensor:
        if not hidden_states:
            raise RuntimeError("MERT returned no hidden states")
        if self.use_weighted_layers:
            if len(hidden_states) != self.layer_logits.numel():
                raise RuntimeError(
                    f"expected {self.layer_logits.numel()} MERT layers, got "
                    f"{len(hidden_states)}"
                )
            weights = self.layer_logits.softmax(dim=0)
            hidden = torch.zeros_like(hidden_states[0])
            for weight, layer_hidden in zip(weights, hidden_states):
                hidden = hidden + weight.to(layer_hidden) * layer_hidden
            return hidden
        return hidden_states[self.selected_layer]

    def encode(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        context = torch.no_grad() if self.backbone_fully_frozen else torch.enable_grad()
        with context:
            output = self.backbone(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = tuple(output.hidden_states)
        hidden = self._combine_layers(hidden_states)
        feature_mask = self._feature_mask(hidden, attention_mask)
        if feature_mask is None:
            return hidden.float().mean(dim=1)
        mask = feature_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        return (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        *,
        return_embeddings: bool = False,
    ):
        embeddings = self.encode(input_values, attention_mask)
        logits = self.classifier(embeddings)
        return (logits, embeddings) if return_embeddings else logits


__all__ = ["MERTArtistClassifier"]
