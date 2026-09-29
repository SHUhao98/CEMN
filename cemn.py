"""Clinical Evidence Memory Network (CEMN).

This file contains only the CEMN plug-in described in the manuscript:

1. Reasoning Anchor (RA) for complementary multimodal rereading.
2. Clinical Evidence Memory Branch (CEMB) for local evidence selection,
   organization, and participant-specific reasoning.
3. The shared sample-adaptive gate used for feature and prediction residuals.
4. A parameter-free memory-level residual readout for case visualization.

The global encoder is deliberately external.  Callers provide its participant-
level feature and logits to :class:`ClinicalEvidenceMemoryNetwork.forward`.
The input modality tensors must already be encoded and aligned at the
question-answer-turn level.

The implementation is consolidated from the DAIC-WOZ and EATD experiment
modules.  The learnable memory slots are latent slots shared across samples;
they are not assigned fixed symptom meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "CEMNConfig",
    "ReasoningAnchor",
    "TurnLevelEvidenceEncoder",
    "ClinicalEvidenceMemoryBranch",
    "AdaptiveGatedResidualFusion",
    "ClinicalEvidenceMemoryNetwork",
    "CEMN",
    "masked_mean",
    "masked_softmax",
]


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Apply softmax over valid entries and return zero at masked entries."""

    mask = mask.to(device=scores.device, dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(masked_scores, dim=dim)
    probabilities = probabilities * mask.to(dtype=scores.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return probabilities / normalizer


def masked_mean(values: Tensor, mask: Tensor, dim: int = 1) -> Tensor:
    """Compute the mean over valid turns."""

    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1e-8)
    return numerator / denominator


@dataclass
class CEMNConfig:
    """Configuration for the dataset-independent CEMN plug-in.

    ``hidden_dim`` is the dimension of every encoded turn representation.
    ``global_dim`` is the dimension of the external global feature.  It defaults
    to ``hidden_dim``.
    """

    hidden_dim: int = 128
    global_dim: Optional[int] = None
    num_classes: int = 2
    side_modalities: Tuple[str, ...] = ("question", "audio", "video")
    num_evidence_queries: int = 6
    num_memory_slots: int = 9
    ra_layers: int = 2
    ra_heads: int = 8
    ra_dropout: float = 0.3
    side_gate_init: float = 0.0
    evidence_gate_bias: float = -3.0
    prediction_dropout: float = 0.35
    ra_gate_init: float = 0.0

    def __post_init__(self) -> None:
        if self.global_dim is None:
            self.global_dim = self.hidden_dim
        if self.hidden_dim <= 0 or self.global_dim <= 0:
            raise ValueError("hidden_dim and global_dim must be positive")
        if self.hidden_dim % self.ra_heads != 0:
            raise ValueError("hidden_dim must be divisible by ra_heads")
        if self.num_evidence_queries <= 0 or self.num_memory_slots <= 0:
            raise ValueError("The numbers of evidence queries and memory slots must be positive")
        if self.ra_layers <= 0:
            raise ValueError("ra_layers must be positive")
        if len(set(self.side_modalities)) != len(self.side_modalities):
            raise ValueError("side_modalities must not contain duplicates")
        if "text" in self.side_modalities:
            raise ValueError("text is the primary modality and must not be in side_modalities")

    @classmethod
    def daic_woz(
        cls,
        *,
        global_dim: Optional[int] = None,
        side_modalities: Sequence[str] = ("question", "audio", "video"),
    ) -> "CEMNConfig":
        """Paper settings for DAIC-WOZ."""

        return cls(
            hidden_dim=128,
            global_dim=global_dim,
            side_modalities=tuple(side_modalities),
            num_evidence_queries=6,
            num_memory_slots=9,
            ra_layers=2,
            ra_heads=8,
            prediction_dropout=0.35,
        )

    @classmethod
    def eatd(
        cls,
        *,
        global_dim: Optional[int] = None,
        side_modalities: Sequence[str] = ("question", "audio"),
    ) -> "CEMNConfig":
        """Paper settings for EATD."""

        return cls(
            hidden_dim=256,
            global_dim=global_dim,
            side_modalities=tuple(side_modalities),
            num_evidence_queries=3,
            num_memory_slots=9,
            ra_layers=2,
            ra_heads=8,
            prediction_dropout=0.5,
        )


def _validate_turn_inputs(
    modalities: Mapping[str, Tensor],
    valid_mask: Tensor,
    side_modalities: Sequence[str],
    hidden_dim: int,
) -> Tuple[int, int]:
    required = ("text", *side_modalities)
    missing = [name for name in required if name not in modalities]
    if missing:
        raise KeyError(f"Missing encoded modalities: {missing}")

    text = modalities["text"]
    if text.ndim != 3:
        raise ValueError("Each modality must have shape [batch, turns, hidden_dim]")
    batch_size, turns, width = text.shape
    if width != hidden_dim:
        raise ValueError(f"text has hidden dimension {width}; expected {hidden_dim}")
    if valid_mask.shape != (batch_size, turns):
        raise ValueError(
            f"valid_mask has shape {tuple(valid_mask.shape)}; expected {(batch_size, turns)}"
        )
    if not torch.all(valid_mask.to(torch.bool).any(dim=1)):
        raise ValueError("Every interview must contain at least one valid turn")

    for name in side_modalities:
        if modalities[name].shape != text.shape:
            raise ValueError(
                f"{name} has shape {tuple(modalities[name].shape)}; "
                f"expected {tuple(text.shape)}"
            )
    return batch_size, turns


class ReasoningAnchor(nn.Module):
    """Complementarily reread encoded turns with an independent RA path.

    The learnable anchor is warm-started for each sample by the masked mean of
    its response-text turns.  It then serves as the query of a stacked
    multi-head attention path over the question, response-text, and available
    behavioral-modality representations.
    """

    def __init__(
        self,
        hidden_dim: int,
        side_modalities: Sequence[str],
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.side_modalities = tuple(side_modalities)
        self.anchor = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.anchor, std=0.02)

        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim, num_heads, dropout=float(dropout), batch_first=True
                )
                for _ in range(num_layers)
            ]
        )
        self.anchor_normalizations = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self, modalities: Mapping[str, Tensor], valid_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        valid_mask = valid_mask.to(device=modalities["text"].device, dtype=torch.bool)
        batch_size, _ = _validate_turn_inputs(
            modalities, valid_mask, self.side_modalities, self.hidden_dim
        )
        context = torch.cat(
            [modalities["text"], *(modalities[name] for name in self.side_modalities)],
            dim=1,
        )
        context_mask = torch.cat(
            [valid_mask for _ in range(1 + len(self.side_modalities))], dim=1
        )
        anchor_state = self.anchor.view(1, 1, -1) + masked_mean(
            modalities["text"], valid_mask
        ).unsqueeze(1)

        layer_attentions = []
        for attention_layer, normalization in zip(
            self.attention_layers, self.anchor_normalizations
        ):
            update, attention = attention_layer(
                anchor_state,
                context,
                context,
                key_padding_mask=~context_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            anchor_state = normalization(anchor_state + update)
            layer_attentions.append(attention)

        anchor_feature = anchor_state.squeeze(1)
        auxiliary = {
            "ra_attention": torch.stack(layer_attentions, dim=1),
            "ra_anchor_feature": anchor_feature,
            "ra_context_mask": context_mask,
        }
        if anchor_feature.shape != (batch_size, self.hidden_dim):
            raise RuntimeError("Unexpected Reasoning Anchor output shape")
        return anchor_feature, auxiliary


class TurnLevelEvidenceEncoder(nn.Module):
    """Construct text-anchored turn evidence from aligned modalities."""

    def __init__(
        self,
        hidden_dim: int,
        side_modalities: Sequence[str],
        *,
        side_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.side_modalities = tuple(side_modalities)
        self.side_projection = nn.ModuleDict(
            {name: nn.Linear(self.hidden_dim, self.hidden_dim) for name in self.side_modalities}
        )
        self.side_scale = nn.ParameterDict(
            {
                name: nn.Parameter(torch.full((self.hidden_dim,), float(side_gate_init)))
                for name in self.side_modalities
            }
        )
        self.normalization = nn.LayerNorm(self.hidden_dim)

    def forward(self, modalities: Mapping[str, Tensor], valid_mask: Tensor) -> Tensor:
        valid_mask = valid_mask.to(device=modalities["text"].device, dtype=torch.bool)
        _validate_turn_inputs(modalities, valid_mask, self.side_modalities, self.hidden_dim)
        turn_evidence = modalities["text"]
        for name in self.side_modalities:
            residual = torch.tanh(self.side_projection[name](modalities[name]))
            turn_evidence = turn_evidence + self.side_scale[name] * residual
        turn_evidence = self.normalization(turn_evidence)
        return turn_evidence * valid_mask.unsqueeze(-1).to(turn_evidence.dtype)


class ClinicalEvidenceMemoryBranch(nn.Module):
    """Select, organize, and aggregate local clinical evidence."""

    def __init__(
        self,
        hidden_dim: int,
        side_modalities: Sequence[str],
        *,
        num_evidence_queries: int,
        num_memory_slots: int,
        side_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_evidence_queries = int(num_evidence_queries)
        self.num_memory_slots = int(num_memory_slots)
        self.turn_encoder = TurnLevelEvidenceEncoder(
            self.hidden_dim, side_modalities, side_gate_init=side_gate_init
        )

        self.evidence_queries = nn.Parameter(
            torch.empty(self.num_evidence_queries, self.hidden_dim)
        )
        nn.init.normal_(self.evidence_queries, std=0.02)
        self.evidence_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.evidence_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.evidence_value = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.latent_memory = nn.Parameter(
            torch.empty(self.num_memory_slots, self.hidden_dim)
        )
        nn.init.normal_(self.latent_memory, std=0.02)
        self.memory_query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.slot_key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.slot_value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.memory_normalization = nn.LayerNorm(self.hidden_dim)

        self.reason_score = nn.Linear(self.hidden_dim, 1)
        self.patient_normalization = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def path_turn_importance(
        evidence_attention: Tensor,
        memory_attention: Tensor,
        reason_attention: Tensor,
    ) -> Tensor:
        """Compute the manuscript's query-memory-reasoning path importance."""

        importance = torch.einsum(
            "bs,bsk,bkt->bt",
            reason_attention,
            memory_attention,
            evidence_attention,
        )
        return importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def forward(
        self, modalities: Mapping[str, Tensor], valid_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        valid_mask = valid_mask.to(device=modalities["text"].device, dtype=torch.bool)
        turn_evidence = self.turn_encoder(modalities, valid_mask)
        batch_size = turn_evidence.shape[0]
        scale = self.hidden_dim**0.5

        query_seed = self.evidence_queries.unsqueeze(0).expand(batch_size, -1, -1)
        evidence_scores = torch.matmul(
            self.evidence_query(query_seed),
            self.evidence_key(turn_evidence).transpose(1, 2),
        ) / scale
        evidence_attention = masked_softmax(
            evidence_scores, valid_mask[:, None, :], dim=-1
        )
        evidence_slots = torch.matmul(
            evidence_attention, self.evidence_value(turn_evidence)
        )

        memory_seed = self.latent_memory.unsqueeze(0).expand(batch_size, -1, -1)
        memory_scores = torch.matmul(
            self.memory_query(memory_seed),
            self.slot_key(evidence_slots).transpose(1, 2),
        ) / scale
        memory_attention = torch.softmax(memory_scores, dim=-1)
        memory_responses = self.memory_normalization(
            memory_seed
            + torch.matmul(memory_attention, self.slot_value(evidence_slots))
        )

        relevance_logits = self.reason_score(memory_responses).squeeze(-1)
        reason_attention = torch.softmax(relevance_logits, dim=-1)
        clinical_feature = self.patient_normalization(
            (reason_attention.unsqueeze(-1) * memory_responses).sum(dim=1)
        )
        turn_importance = self.path_turn_importance(
            evidence_attention, memory_attention, reason_attention
        )

        auxiliary = {
            "turn_evidence": turn_evidence,
            "evidence_slots": evidence_slots,
            "evidence_attention": evidence_attention,
            "memory_responses": memory_responses,
            "memory_attention": memory_attention,
            "memory_relevance_logits": relevance_logits,
            "reason_attention": reason_attention,
            "turn_importance": turn_importance,
        }
        return clinical_feature, auxiliary


class AdaptiveGatedResidualFusion(nn.Module):
    """Fuse RA, CEMB, and an external global prediction with one shared gate."""

    def __init__(
        self,
        hidden_dim: int,
        global_dim: int,
        num_classes: int,
        *,
        gate_bias: float = -3.0,
        prediction_dropout: float = 0.5,
        ra_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.global_dim = int(global_dim)
        self.num_classes = int(num_classes)

        self.ra_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.global_dim),
            nn.GELU(),
        )
        self.ra_gate_logit = nn.Parameter(torch.tensor(float(ra_gate_init)))
        self.local_projection = nn.Linear(self.hidden_dim, self.global_dim)

        self.evidence_gate = nn.Linear(self.global_dim + self.hidden_dim, 1)
        nn.init.normal_(self.evidence_gate.weight, std=1e-3)
        nn.init.constant_(self.evidence_gate.bias, float(gate_bias))

        self.prediction_residual = nn.Sequential(
            nn.Dropout(float(prediction_dropout)),
            nn.Linear(self.global_dim, self.num_classes),
        )
        nn.init.zeros_(self.prediction_residual[1].weight)
        nn.init.zeros_(self.prediction_residual[1].bias)

    def memory_level_readout(
        self, memory_responses: Tensor
    ) -> Dict[str, Tensor]:
        """Compute memory-level residual readouts for case visualization.

        The readout reuses the trained local projection and the linear
        classifier of the residual prediction head. It introduces no
        additional parameters or training objective.
        """

        memory_projected = self.local_projection(memory_responses)

        # Case visualizations are generated with model.eval(). In evaluation
        # mode, this linear layer is equivalent to the complete residual head
        # because its preceding dropout layer is disabled.
        memory_residual_logits = self.prediction_residual[1](memory_projected)
        memory_probabilities = torch.softmax(memory_residual_logits, dim=-1)
        memory_polarity = (
            memory_probabilities[..., 1] - memory_probabilities[..., 0]
        )

        return {
            "memory_residual_logits": memory_residual_logits,
            "memory_probabilities": memory_probabilities,
            "memory_polarity": memory_polarity,
        }

    def forward(
        self,
        global_feature: Tensor,
        global_logits: Tensor,
        ra_feature: Tensor,
        clinical_feature: Tensor,
        memory_responses: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if global_feature.ndim != 2 or global_feature.shape[-1] != self.global_dim:
            raise ValueError(
                f"global_feature must have shape [batch, {self.global_dim}]"
            )
        if global_logits.shape != (global_feature.shape[0], self.num_classes):
            raise ValueError(
                f"global_logits must have shape [batch, {self.num_classes}]"
            )
        expected_local_shape = (global_feature.shape[0], self.hidden_dim)
        if ra_feature.shape != expected_local_shape:
            raise ValueError(f"ra_feature must have shape {expected_local_shape}")
        if clinical_feature.shape != expected_local_shape:
            raise ValueError(f"clinical_feature must have shape {expected_local_shape}")
        if (
            memory_responses.ndim != 3
            or memory_responses.shape[0] != global_feature.shape[0]
            or memory_responses.shape[-1] != self.hidden_dim
        ):
            raise ValueError(
                "memory_responses must have shape "
                f"[batch, num_memory_slots, {self.hidden_dim}]"
            )

        ra_strength = torch.sigmoid(self.ra_gate_logit)
        enhanced_global = global_feature + ra_strength * self.ra_projection(ra_feature)

        gate = torch.sigmoid(
            self.evidence_gate(torch.cat([enhanced_global, clinical_feature], dim=-1))
        )
        feature_residual = gate * self.local_projection(clinical_feature)
        fused_feature = enhanced_global + feature_residual

        raw_prediction_residual = self.prediction_residual(fused_feature)
        prediction_residual = gate * raw_prediction_residual
        final_logits = global_logits + prediction_residual
        memory_readout = self.memory_level_readout(memory_responses)

        auxiliary = {
            "ra_strength": ra_strength,
            "enhanced_global_feature": enhanced_global,
            "gate": gate,
            "feature_residual": feature_residual,
            "fused_feature": fused_feature,
            "raw_prediction_residual": raw_prediction_residual,
            "prediction_residual": prediction_residual,
            **memory_readout,
        }
        return final_logits, auxiliary


class ClinicalEvidenceMemoryNetwork(nn.Module):
    """Backbone-agnostic CEMN containing RA, CEMB, and gated fusion only.

    Parameters passed to ``forward``
    --------------------------------
    modalities:
        Mapping from modality name to encoded turn tensor ``[B, T, H]``.
        ``text`` is required and represents participant response text.  Every
        configured side modality must use the same shape and turn alignment.
    valid_mask:
        Boolean or binary tensor ``[B, T]``.  Attention is normalized only over
        entries marked as valid.
    global_feature:
        Participant-level representation ``[B, Dg]`` from any external global
        encoder.
    global_logits:
        Backbone prediction logits ``[B, C]`` from that external encoder.
    """

    def __init__(self, config: Optional[CEMNConfig] = None) -> None:
        super().__init__()
        self.config = config or CEMNConfig()
        config = self.config

        self.reasoning_anchor = ReasoningAnchor(
            config.hidden_dim,
            config.side_modalities,
            num_layers=config.ra_layers,
            num_heads=config.ra_heads,
            dropout=config.ra_dropout,
        )
        self.cemb = ClinicalEvidenceMemoryBranch(
            config.hidden_dim,
            config.side_modalities,
            num_evidence_queries=config.num_evidence_queries,
            num_memory_slots=config.num_memory_slots,
            side_gate_init=config.side_gate_init,
        )
        self.fusion = AdaptiveGatedResidualFusion(
            config.hidden_dim,
            int(config.global_dim),
            config.num_classes,
            gate_bias=config.evidence_gate_bias,
            prediction_dropout=config.prediction_dropout,
            ra_gate_init=config.ra_gate_init,
        )

    def forward(
        self,
        modalities: Mapping[str, Tensor],
        valid_mask: Tensor,
        global_feature: Tensor,
        global_logits: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        ra_feature, ra_auxiliary = self.reasoning_anchor(modalities, valid_mask)
        clinical_feature, cemb_auxiliary = self.cemb(modalities, valid_mask)
        final_logits, fusion_auxiliary = self.fusion(
            global_feature,
            global_logits,
            ra_feature,
            clinical_feature,
            cemb_auxiliary["memory_responses"],
        )

        auxiliary = {
            "global_feature": global_feature,
            "global_logits": global_logits,
            "ra_feature": ra_feature,
            "clinical_feature": clinical_feature,
            **ra_auxiliary,
            **cemb_auxiliary,
            **fusion_auxiliary,
        }
        return final_logits, auxiliary


CEMN = ClinicalEvidenceMemoryNetwork
