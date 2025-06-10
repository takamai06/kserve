from typing import List, Set
import torch
from transformers import LogitsProcessor
import logging

logger = logging.getLogger(__name__)


class RestrictToTokensProcessor(LogitsProcessor):
    """
    Restricts the model's output to a predefined set of allowed tokens.
    This implementation is compatible with vLLM's single-sequence processing loop.
    """
    def __init__(self, allowed_token_ids: List[int]):
        logger.info(f"RestrictToTokensProcessor initialized with allowed_token_ids: {allowed_token_ids}")
        self.allowed_token_ids: Set[int] = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids (`torch.LongTensor`): A 1D tensor of shape `(sequence_length,)`.
            scores (`torch.FloatTensor`): A 1D tensor of shape `(vocab_size,)`.
        """
        mask = torch.full_like(scores, float("-inf"))

        # Create a tensor of the allowed token indices for efficient indexing
        allowed_indices = torch.tensor(list(self.allowed_token_ids), device=scores.device, dtype=torch.long)

        # Copy the original scores for the allowed tokens into the mask
        mask.scatter_(0, allowed_indices, scores.index_select(0, allowed_indices))
        
        return mask