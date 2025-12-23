"""
Probe models for detecting attributes from LLM hidden states.
"""

import torch
import torch.nn.functional as F
from torch import nn


class LinearProbeClassification(nn.Module):
    """
    Linear probe for classification from hidden states.

    Used for both reading probes (detect attributes) and control probes (for steering).
    """
    def __init__(self, device, probe_class, input_dim=4096, logistic=True):
        super().__init__()
        self.input_dim = input_dim
        self.probe_class = probe_class

        if logistic:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
                nn.Sigmoid()
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.input_dim, self.probe_class),
            )

        self.apply(self._init_weights)
        self.to(device)

    def forward(self, act, y=None):
        """
        Forward pass.

        Args:
            act: Hidden state activations [batch, hidden_dim]
            y: Optional labels for computing loss [batch]

        Returns:
            logits: Class logits [batch, num_classes]
            loss: Cross-entropy loss (if y provided)
        """
        logits = self.proj(act)

        if y is None:
            return logits, None
        else:
            targets = y.to(torch.long)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            return logits, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class TrainerConfig:
    """Configuration for probe training optimizer."""
    learning_rate = 1e-3
    betas = (0.9, 0.95)
    weight_decay = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)