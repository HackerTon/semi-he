import torch
from typing import Tuple, Iterator


class EMA:
    def __init__(self, model: torch.nn.Module, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class EMAFunction:
    def update_parameter_to_model(
        self,
        decay: float,
        model: torch.nn.Module,
        named_parameters: Iterator[tuple[str, torch.nn.Parameter]],
    ):
        for parameters_a, parameters_b in zip(
            model.named_parameters(),
            named_parameters,
        ):
            a_name, a_param = parameters_a
            b_name, b_param = parameters_b
            new_average = (1.0 - decay) * a_param.data + decay * b_param.data
            a_param.data = new_average
