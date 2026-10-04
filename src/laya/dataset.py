import torch
import torch.nn as nn
import random
import copy

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

    expected_option_counts = torch.tensor([len(x["options"]) for x in examples])
    actual_option_counts = option_marker_mask.sum(dim=1)
    if not torch.equal(expected_option_counts, actual_option_counts.cpu()):
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


# raw exmaples
LABELS = [
    "Billing",
    "Technical support",
    "Sales",
    "Fraud review",
    "Account support",
    "Refunds",
]

raw_examples = [
    # -------------------------
    # Billing
    # -------------------------
    {
        "context": "The customer was charged twice for the same monthly subscription.",
        "correct_label": "Billing",
    },
    {
        "context": "The invoice shows a higher amount than the price listed in the contract.",
        "correct_label": "Billing",
    },
    {
        "context": "The customer says an unexpected service fee appeared on their latest statement.",
        "correct_label": "Billing",
    },
    {
        "context": "The company received an invoice but the tax amount appears to be incorrect.",
        "correct_label": "Billing",
    },
    {
        "context": "The customer's credit card was charged after they had already paid by bank transfer.",
        "correct_label": "Billing",
    },
    {
        "context": "A customer wants clarification about several line items on their monthly invoice.",
        "correct_label": "Billing",
    },
    {
        "context": "The customer says the amount charged does not match the plan they selected.",
        "correct_label": "Billing",
    },
    {
        "context": "The finance department says the invoice number on the payment request is incorrect.",
        "correct_label": "Billing",
    },
    {
        "context": "The customer was billed for an add-on they say they never purchased.",
        "correct_label": "Billing",
    },
    {
        "context": "A business customer wants to understand why this month's invoice increased.",
        "correct_label": "Billing",
    },
    # -------------------------
    # Technical support
    # -------------------------
    {
        "context": "The mobile application crashes immediately after the user signs in.",
        "correct_label": "Technical support",
    },
    {
        "context": "The website returns an internal server error whenever the customer uploads a file.",
        "correct_label": "Technical support",
    },
    {
        "context": "The API started returning timeout errors after the latest software update.",
        "correct_label": "Technical support",
    },
    {
        "context": "The user cannot connect the desktop application to the cloud service.",
        "correct_label": "Technical support",
    },
    {
        "context": "The customer reports that the dashboard remains blank after loading.",
        "correct_label": "Technical support",
    },
    {
        "context": "A user says notifications stopped appearing after they upgraded the app.",
        "correct_label": "Technical support",
    },
    {
        "context": "The integration fails whenever the customer tries to authenticate with OAuth.",
        "correct_label": "Technical support",
    },
    {
        "context": "The customer sees corrupted characters when exporting a report to CSV.",
        "correct_label": "Technical support",
    },
    {
        "context": "A customer cannot install the latest desktop client on their computer.",
        "correct_label": "Technical support",
    },
    {
        "context": "The application becomes unresponsive whenever the user opens the analytics page.",
        "correct_label": "Technical support",
    },
    # -------------------------
    # Sales
    # -------------------------
    {
        "context": "A company wants pricing for five hundred enterprise user licenses.",
        "correct_label": "Sales",
    },
    {
        "context": "A prospective customer wants a demonstration before purchasing the product.",
        "correct_label": "Sales",
    },
    {
        "context": "A company wants to negotiate a multi-year enterprise contract.",
        "correct_label": "Sales",
    },
    {
        "context": "A startup is asking whether there is a discounted annual pricing plan.",
        "correct_label": "Sales",
    },
    {
        "context": "A large organization wants a quote for deploying the product across multiple offices.",
        "correct_label": "Sales",
    },
    {
        "context": "A potential customer wants to compare the professional and enterprise plans.",
        "correct_label": "Sales",
    },
    {
        "context": "A company asks whether volume discounts are available for a large number of seats.",
        "correct_label": "Sales",
    },
    {
        "context": "A customer is interested in upgrading from an individual plan to an enterprise contract.",
        "correct_label": "Sales",
    },
    {
        "context": "A prospective client wants to discuss custom pricing and contract terms.",
        "correct_label": "Sales",
    },
    {
        "context": "A business wants a formal quote before presenting the software purchase to management.",
        "correct_label": "Sales",
    },
    # -------------------------
    # Fraud review
    # -------------------------
    {
        "context": "Several purchases were made from locations the customer has never visited.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The account suddenly made multiple unusually large transactions within a few minutes.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The customer says they do not recognize several recent purchases on their account.",
        "correct_label": "Fraud review",
    },
    {
        "context": "A transaction was attempted shortly after the account password was changed from an unfamiliar device.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The system detected repeated purchases using several different payment cards from the same account.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The customer reports that someone may have gained unauthorized access to their payment account.",
        "correct_label": "Fraud review",
    },
    {
        "context": "A newly created account immediately attempted a very high value transaction.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The same payment method was used from multiple distant geographic locations within one hour.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The transaction pattern is significantly different from the customer's normal activity.",
        "correct_label": "Fraud review",
    },
    {
        "context": "The customer claims that several transactions were made after their device was stolen.",
        "correct_label": "Fraud review",
    },
    # -------------------------
    # Account support
    # -------------------------
    {
        "context": "The customer forgot their password and cannot access the account.",
        "correct_label": "Account support",
    },
    {
        "context": "The user changed phone numbers and can no longer receive the login verification code.",
        "correct_label": "Account support",
    },
    {
        "context": "The customer wants to update the email address associated with the account.",
        "correct_label": "Account support",
    },
    {
        "context": "A user says their account was locked after too many failed login attempts.",
        "correct_label": "Account support",
    },
    {
        "context": "The customer wants to change the name displayed on their profile.",
        "correct_label": "Account support",
    },
    {
        "context": "The user can no longer access the email address used for account recovery.",
        "correct_label": "Account support",
    },
    {
        "context": "The customer wants to enable two-factor authentication on their account.",
        "correct_label": "Account support",
    },
    {
        "context": "The user accidentally created two accounts and wants help resolving the duplicate.",
        "correct_label": "Account support",
    },
    {
        "context": "The customer wants to close their account permanently.",
        "correct_label": "Account support",
    },
    {
        "context": "The user needs help changing the security settings associated with their profile.",
        "correct_label": "Account support",
    },
    # -------------------------
    # Refunds
    # -------------------------
    {
        "context": "The customer canceled the purchase and wants the payment returned.",
        "correct_label": "Refunds",
    },
    {
        "context": "A customer returned the product but has not yet received their money back.",
        "correct_label": "Refunds",
    },
    {
        "context": "The customer wants their subscription payment reversed after canceling during the trial period.",
        "correct_label": "Refunds",
    },
    {
        "context": "The user says the merchant approved a refund but the funds have not appeared yet.",
        "correct_label": "Refunds",
    },
    {
        "context": "A customer purchased the wrong plan and wants the transaction reversed.",
        "correct_label": "Refunds",
    },
    {
        "context": "The customer canceled an order before shipment and is asking when the money will be returned.",
        "correct_label": "Refunds",
    },
    {
        "context": "A user was charged for a service they immediately canceled and is requesting the money back.",
        "correct_label": "Refunds",
    },
    {
        "context": "The customer says they were promised a refund several days ago but have not received it.",
        "correct_label": "Refunds",
    },
    {
        "context": "The customer returned an item and wants to know the status of the reimbursement.",
        "correct_label": "Refunds",
    },
    {
        "context": "A user wants a payment reversed because they accidentally purchased the same item twice.",
        "correct_label": "Refunds",
    },
]


def build_decision_examples(raw_examples, labels, seed=42):
    rng = random.Random(seed)

    examples = []

    for raw in raw_examples:
        correct_label = raw["correct_label"]

        distractors = [x for x in labels if x != correct_label]

        n_options = rng.randint(3, len(labels))

        selected_distractors = rng.sample(distractors, n_options - 1)

        options = [correct_label, *selected_distractors]

        rng.shuffle(options)

        target = options.index(correct_label)
        examples.append(
            {
                "context": raw["context"],
                "question": "which team should handle this request?",
                "options": options,
                "target": target,
            }
        )
    return examples


def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}
