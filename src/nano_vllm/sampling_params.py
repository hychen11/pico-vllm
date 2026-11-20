"""Generation settings for one request.

This module describes how tokens should be sampled and when generation should stop.
It stores user choices and validates them; the sampler and engine apply those choices.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Options supplied when creating a text-generation request."""

    # 0 means greedy (always the highest-probability token, deterministic);
    # > 0 samples from softmax(logits / temperature).
    temperature: float = 1.0
    # Maximum number of *generated* tokens; the prompt does not count.
    max_tokens: int = 128
    # Keep generating past EOS until max_tokens (useful for benchmarks).
    ignore_eos: bool = False

    def __post_init__(self):
        """Validate field values after the dataclass initializer assigns them."""
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
