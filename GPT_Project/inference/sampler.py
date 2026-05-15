import torch


def top_k_top_p_filtering(
    logits,
    top_k=50,
    top_p=0.9
):

    # ============================================
    # TOP-K
    # ============================================

    if top_k > 0:

        values, _ = torch.topk(
            logits,
            top_k
        )

        min_values = values[:, -1].unsqueeze(-1)

        logits = torch.where(
            logits < min_values,
            torch.full_like(logits, float("-inf")),
            logits
        )

    # ============================================
    # TOP-P
    # ============================================

    if top_p < 1.0:

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True
        )

        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1
        )

        sorted_indices_to_remove = cumulative_probs > top_p

        sorted_indices_to_remove[:, 1:] = \
            sorted_indices_to_remove[:, :-1].clone()

        sorted_indices_to_remove[:, 0] = 0

        for batch_idx in range(logits.size(0)):

            remove_indices = sorted_indices[
                batch_idx,
                sorted_indices_to_remove[batch_idx]
            ]

            logits[batch_idx, remove_indices] = float("-inf")

    return logits


def apply_repetition_penalty(
    logits,
    generated_tokens,
    penalty=1.15
):

    if generated_tokens is None:
        return logits

    unique_tokens = set(generated_tokens)

    for token in unique_tokens:
        logits[:, token] /= penalty

    return logits


def sample(
    logits,
    generated_tokens=None,
    temperature=0.8,
    top_k=40,
    top_p=0.92
):

    # ============================================
    # TEMPERATURE
    # ============================================

    logits = logits / temperature

    # ============================================
    # REPETITION PENALTY
    # ============================================

    logits = apply_repetition_penalty(
        logits,
        generated_tokens
    )

    # ============================================
    # FILTERING
    # ============================================

    logits = top_k_top_p_filtering(
        logits,
        top_k=top_k,
        top_p=top_p
    )

    probs = torch.softmax(
        logits,
        dim=-1
    )

    next_token = torch.multinomial(
        probs,
        num_samples=1
    )

    return next_token