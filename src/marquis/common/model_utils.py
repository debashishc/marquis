import torch
import torch.nn.functional as F
from transformers import Qwen2_5OmniProcessor


def extract_score(
    logits: torch.Tensor,
    processor: Qwen2_5OmniProcessor,
    new_token_num: int = 100,
    new_token_prefix: str = "<CON_{idx}>",
) -> float:
    # Model outputs are typically [batch, seq, vocab]. We want the logits for
    # the final generated position over the vocabulary, not the final batch row
    # over the sequence axis.
    if logits.ndim == 3:
        logits = logits[-1, -1, :]
    elif logits.ndim == 2:
        logits = logits[-1, :]
    else:
        raise ValueError(f"Unsupported logits rank for score extraction: {tuple(logits.shape)}")

    conf_tokens = [new_token_prefix.format(idx=idx) for idx in range(new_token_num)]
    conf_token_ids = processor.tokenizer.convert_tokens_to_ids(conf_tokens)
    vocab_size = logits.shape[-1]
    bad_ids = [
        tok_id for tok_id in conf_token_ids if tok_id is None or tok_id < 0 or tok_id >= vocab_size
    ]
    if bad_ids:
        raise ValueError(
            f"Confidence token ids out of range for logits vocab size {vocab_size}: "
            f"{bad_ids[:5]}{'...' if len(bad_ids) > 5 else ''}"
        )

    conf_logits = logits[conf_token_ids]
    logprob = F.log_softmax(conf_logits, dim=-1)
    probs = torch.exp(logprob)
    scale_values = torch.linspace(0.0, 1.0, new_token_num).to(logits.device)
    return (probs * scale_values).sum().item()
