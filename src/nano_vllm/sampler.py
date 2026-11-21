"""Turn each sequence's last-token logits into one sampled token id (M5).

``ModelRunner.run`` returns logits ``[num_seqs, vocab]``; the engine passes them
here together with each sequence's temperature.
"""

from __future__ import annotations

import torch


def sample(
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        generator: torch.Generator | None = None,
) -> list[int]:
    """Pick one token per row.

    logits: ``[num_seqs, vocab]``; temperatures: ``[num_seqs]`` (float).
    - temperature == 0: greedy, ``argmax`` of the row;
    - temperature > 0: softmax(logits / temperature), then draw one token
      (``torch.multinomial``, using ``generator`` so tests are reproducible).
    A batch may mix greedy and sampled rows.
    Returns one token id per row, in row order.
    """
    logits = logits.float()  # bf16/fp16 models: softmax and multinomial in fp32
    greedy = torch.argmax(logits, dim=-1)
    t = torch.where(temperatures == 0, 1, temperatures)
    probs = torch.softmax(logits / t[:, None], dim=-1)
    sampled = torch.multinomial(probs, 1, generator=generator).squeeze(1)
    return torch.where(temperatures == 0, greedy, sampled).tolist()
