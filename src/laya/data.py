import torch


def build_decision_sequence(
    state_tokens: list[str],
    question_tokens: list[str],
    options: list[str],
):
    sequence = [
        "[STATE]",
        *state_tokens,
        "[QUESTION]",
        *question_tokens,
        "[OPTIONS]",
    ]
    for op in options:
        sequence.extend(
            [
                "[MASK]",
                op,
            ]
        )
    return sequence


def encode_batch(
    sequences: list[list[str]],
    token_to_id: dict[str, int],
    pad_token: str = "[PAD]",
):
    max_length = max(len(sq) for sq in sequences)

    pad_id = token_to_id[pad_token]

    rows = []
    for sq in sequences:
        token_ids = [token_to_id[t] for t in sq]

        padding = [pad_id] * (max_length - len(token_ids))

        rows.append(token_ids + padding)
    return torch.tensor(rows, dtype=torch.long)
