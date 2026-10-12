from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from .pretrained import PretrainedDecisionModel
from .dataset import encode_decision_batch
from .multi_objective_reward import (
    expected_cost_decision,
)


class DecisionPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        temperature=1.0,
        device="cpu",
    ):
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.temperature = float(temperature)

        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")

    @torch.no_grad()
    def predict_proba(self, examples):
        batch = encode_decision_batch(
            examples,
            self.tokenizer,
        )

        batch = {key: value.to(self.device) for key, value in batch.items()}

        logits, _, valid_option_mask = self.model(**batch)

        calibrated_logits = logits / self.temperature

        probabilities = F.softmax(
            calibrated_logits,
            dim=-1,
        )

        results = []

        for i, example in enumerate(examples):
            n_options = len(example["options"])

            if int(valid_option_mask[i].sum()) != n_options:
                raise ValueError(
                    "Option marker count does not match " "the number of options."
                )

            results.append(probabilities[i, :n_options].cpu().tolist())

        return results

    def predict(
        self,
        examples,
        confidence_threshold=0.75,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be in [0, 1].")
        probabilities = self.predict_proba(examples)

        results = []

        for example, proba in zip(
            examples,
            probabilities,
        ):
            confidence = max(proba)

            prediction = proba.index(confidence)

            review = confidence < confidence_threshold

            results.append(
                {
                    "probabilities": dict(
                        zip(
                            example["options"],
                            proba,
                        )
                    ),
                    "prediction": example["options"][prediction],
                    "confidence": confidence,
                    "action": ("review" if review else "auto"),
                }
            )

        return results

    def predict_with_cost(
        self,
        examples,
        cost_matrix,
        review_cost=0.3,
    ):
        probabilities = self.predict_proba(examples)

        results = []

        for example, proba in zip(
            examples,
            probabilities,
        ):
            p = torch.tensor(
                proba,
                dtype=torch.float32,
            )

            costs = torch.as_tensor(
                cost_matrix,
                dtype=torch.float32,
            )

            n_options = len(proba)

            if costs.shape != (
                n_options,
                n_options,
            ):
                raise ValueError("Cost matrix must match option count.")

            decision = expected_cost_decision(
                p,
                costs,
                review_cost=review_cost,
            )

            prediction = int(decision["best_action"].item())

            review = bool(decision["should_review"].item())

            results.append(
                {
                    "probabilities": dict(
                        zip(
                            example["options"],
                            proba,
                        )
                    ),
                    "prediction": example["options"][prediction],
                    "expected_cost": float(decision["expected_cost"].item()),
                    "action": ("review" if review else "auto"),
                }
            )

        return results

    def save(
        self,
        path,
        model_name,
        decision_hidden_dim,
    ):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_name": model_name,
            "decision_hidden_dim": decision_hidden_dim,
            "model_state_dict": (self.model.state_dict()),
            "temperature": self.temperature,
        }

        torch.save(
            checkpoint,
            path,
        )

    @classmethod
    def load(
        cls,
        path,
        device="cpu",
    ):
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model_name = checkpoint["model_name"]

        decision_hidden_dim = checkpoint["decision_hidden_dim"]

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = PretrainedDecisionModel(
            model_name=model_name,
            decision_hidden_dim=decision_hidden_dim,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            temperature=checkpoint["temperature"],
            device=device,
        )
