"""Clinical Evidence Memory Network (CEMN).

This file contains only the CEMN plug-in described in the manuscript:

1. Reasoning Anchor (RA) for complementary multimodal rereading.
2. Clinical Evidence Memory Branch (CEMB) for local evidence selection,
   organization, and participant-specific reasoning.
3. The shared sample-adaptive gate used for feature and prediction residuals.

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
from torch.nn import functional as F

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


class _GraphAttentionHead(nn.Module):
    """Dense additive graph-attention head used by the experimental RA."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = float(dropout)
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.source_attention = nn.Parameter(torch.empty(output_dim, 1))
        self.target_attention = nn.Parameter(torch.empty(output_dim, 1))
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.source_attention, gain=1.414)
        nn.init.xavier_uniform_(self.target_attention, gain=1.414)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tuple[Tensor, Tensor]:
        projected = torch.matmul(features, self.weight)
        scores = self.activation(
            projected @ self.source_attention
            + (projected @ self.target_attention).transpose(1, 2)
        )
        attention = masked_softmax(scores, adjacency, dim=-1)
        dropped_attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.matmul(dropped_attention, projected)
        return F.elu(output), attention


class _MultiHeadGraphAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = hidden_dim // num_heads
        self.dropout = float(dropout)
        self.heads = nn.ModuleList(
            [_GraphAttentionHead(hidden_dim, head_dim, dropout) for _ in range(num_heads)]
        )

    def forward(self, features: Tensor, adjacency: Tensor) -> Tuple[Tensor, Tensor]:
        head_outputs = []
        head_attentions = []
        for head in self.heads:
            output, attention = head(features, adjacency)
            head_outputs.append(output)
            head_attentions.append(attention)
        output = torch.cat(head_outputs, dim=-1)
        output = F.dropout(output, self.dropout, training=self.training)
        return output, torch.stack(head_attentions, dim=1)


class ReasoningAnchor(nn.Module):
    """Complementarily reread encoded turns with an independent RA path.

    The learnable anchor is warm-started for each sample by the masked mean of
    its response-text turns.  In the experimental realization, the anchor and
    modality nodes interact through stacked multi-head graph attention.  The
    anchor is bidirectionally connected to every valid node; side-to-text edges
    retain turn alignment and temporal edges connect adjacent response turns.
    """

    def __init__(
        self,
        hidden_dim: int,
        side_modalities: Sequence[str],
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        side_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.side_modalities = tuple(side_modalities)
        self.anchor = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.anchor, std=0.02)

        self.side_projection = nn.ModuleDict(
            {name: nn.Linear(self.hidden_dim, self.hidden_dim) for name in self.side_modalities}
        )
        self.side_scale = nn.ParameterDict(
            {
                name: nn.Parameter(torch.full((self.hidden_dim,), float(side_gate_init)))
                for name in self.side_modalities
            }
        )
        self.node_normalization = nn.LayerNorm(self.hidden_dim)
        self.attention_layers = nn.ModuleList(
            [
                _MultiHeadGraphAttention(self.hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def _adjacency(self, valid_mask: Tensor) -> Tensor:
        batch_size, turns = valid_mask.shape
        num_modality_blocks = 1 + len(self.side_modalities)
        local_nodes = num_modality_blocks * turns
        total_nodes = local_nodes + 1
        anchor_index = local_nodes
        device = valid_mask.device

        adjacency = torch.zeros(total_nodes, total_nodes, dtype=torch.bool, device=device)
        text_indices = torch.arange(turns, device=device)
        if turns > 1:
            adjacency[text_indices[:-1], text_indices[1:]] = True
            adjacency[text_indices[1:], text_indices[:-1]] = True

        for block_index in range(len(self.side_modalities)):
            side_indices = torch.arange(turns, device=device) + (block_index + 1) * turns
            adjacency[text_indices, side_indices] = True

        adjacency[anchor_index, :] = True
        adjacency[:, anchor_index] = True
        adjacency.fill_diagonal_(True)
        adjacency = adjacency.unsqueeze(0).expand(batch_size, -1, -1)

        local_mask = torch.cat([valid_mask for _ in range(num_modality_blocks)], dim=1)
        node_mask = torch.cat(
            [local_mask, torch.ones(batch_size, 1, dtype=torch.bool, device=device)], dim=1
        )
        adjacency = adjacency & node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        return adjacency | torch.eye(total_nodes, dtype=torch.bool, device=device).unsqueeze(0)

    def forward(
        self, modalities: Mapping[str, Tensor], valid_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        valid_mask = valid_mask.to(device=modalities["text"].device, dtype=torch.bool)
        batch_size, _ = _validate_turn_inputs(
            modalities, valid_mask, self.side_modalities, self.hidden_dim
        )
        turn_mask = valid_mask.unsqueeze(-1).to(dtype=modalities["text"].dtype)

        node_blocks = [modalities["text"] * turn_mask]
        for name in self.side_modalities:
            side = self.side_scale[name] * torch.tanh(
                self.side_projection[name](modalities[name])
            )
            node_blocks.append(side * turn_mask)

        anchor_seed = self.anchor.view(1, 1, -1) + masked_mean(
            modalities["text"], valid_mask
        ).unsqueeze(1)
        nodes = self.node_normalization(torch.cat([*node_blocks, anchor_seed], dim=1))
        adjacency = self._adjacency(valid_mask)

        layer_attentions = []
        for attention_layer in self.attention_layers:
            update, attention = attention_layer(nodes, adjacency)
            nodes = nodes + update
            layer_attentions.append(attention)

        anchor_feature = nodes[:, -1, :]
        auxiliary = {
            "ra_attention": torch.stack(layer_attentions, dim=1),
            "ra_anchor_feature": anchor_feature,
            "ra_adjacency": adjacency,
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
        self.severity_head = nn.Linear(self.hidden_dim, 1)
        self.patient_normalization = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def evidence_diversity_loss(evidence_attention: Tensor) -> Tensor:
        """Penalize overlap among Evidence Query attention distributions."""

        gram = torch.bmm(evidence_attention, evidence_attention.transpose(1, 2))
        query_count = gram.shape[1]
        diagonal = torch.eye(
            query_count, device=gram.device, dtype=gram.dtype
        ).unsqueeze(0) * gram
        off_diagonal = gram - diagonal
        return off_diagonal.square().sum((1, 2)).mean() / max(
            1, query_count * (query_count - 1)
        )

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

        severity = self.severity_head(memory_responses).squeeze(-1)
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
            "severity": severity,
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

    def forward(
        self,
        global_feature: Tensor,
        global_logits: Tensor,
        ra_feature: Tensor,
        clinical_feature: Tensor,
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

        auxiliary = {
            "ra_strength": ra_strength,
            "enhanced_global_feature": enhanced_global,
            "gate": gate,
            "feature_residual": feature_residual,
            "fused_feature": fused_feature,
            "raw_prediction_residual": raw_prediction_residual,
            "prediction_residual": prediction_residual,
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
            side_gate_init=config.side_gate_init,
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
            global_feature, global_logits, ra_feature, clinical_feature
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
