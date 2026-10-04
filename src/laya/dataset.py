import torch
import torch.nn as nn

examples = [
    {
        "context": "The customer was charged twice.",
        "question": "Which action should we take?",
        "options": [
            "Send to billing",
            "Issue an automatic refund",
        ],
        "target": 0,
    },
    {
        "context": "The application crashes after login.",
        "question": "Which action should we take?",
        "options": [
            "Send to billing",
            "Send to technical support",
            "Escalate to engineering",
            "Ask the customer for more information",
        ],
        "target": 1,
    },
    {
        "context": "A company wants enterprise pricing.",
        "question": "Which action should we take?",
        "options": [
            "Send to billing",
            "Send to technical support",
            "Send to sales",
        ],
        "target": 2,
    },
]


def format_decision_example(
    context: str, question: str, options: list[str], marker: str
):
    parts = [context, question]

    for op in options:
        parts.append(f"{marker} {op}")

    return "\n".join(parts)


def encode_decision_batch(examples, tokenizer):
    texts = [
        format_decision_example(
            context=x["context"],
            question=x["question"],
            options=x["options"],
            marker=tokenizer.mask_token,
        )
        for x in examples
    ]

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    option_marker_mask = encoded["input_ids"] == tokenizer.mask_token_id

    expected_option_counts = torch.tensor([len(x["options"] for x in examples)])
    actual_option_counts = option_marker_mask.sum(dim=1)
    if not torch.equal(expected_option_counts, actual_option_counts.cput()):
        raise ValueError("One or more option markers " "were lost during tokenization.")

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "option_marker_mask": option_marker_mask,
    }


def gather_option_states(hidden_states, option_marker_mask):
    batch_size = hidden_states.shape[0]

    option_counts = option_marker_mask.sum(dim=1)

    max_options = int(option_counts.max().item())

    hidden_size = hidden_states.shape[-1]

    option_states = hidden_states.new_zeros(batch_size, max_options, hidden_size)

    valid_option_mask = torch.zeros(
        batch_size, max_options, dtype=torch.bool, device=hidden_states.device
    )

    for batch_idx in range(batch_size):
        states = hidden_states[batch_idx, option_marker_mask[batch_idx]]
        n_options = states.shape[0]

        option_states[batch_idx, :n_options] = states
        valid_option_mask[batch_idx, :n_options] = True

    return option_states, valid_option_mask
