from typing import Tuple

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2

from src.dataloader.dataset.he_dataset_direct import HeDataset
from src.dataloader.transform import ImagenetNormalize, ToNormalized
from src.experiment.experimentbase import ExperimentBase
from src.model.model import UNETNetwork
from src.service.parameter import Parameter


class HeExperiment(ExperimentBase):
    def __init__(self, parameter: Parameter) -> None:
        super().__init__()

        self.train_dataloader, self.test_dataloader = create_dataloader(
            path=parameter.data_path,
            batch_size=parameter.batch_size_train,
        )

        self.model = UNETNetwork(numberClass=3)
        self.model_teacher = UNETNetwork(numberClass=3)
        self.preprocessor = v2.Compose(
            [
                ToNormalized(),
                ImagenetNormalize(),
            ]
        )

        # Move weights to specified device
        # Load model
        weights = torch.load(parameter.pretrain_path)
        self.model.load_state_dict(weights)
        self.model = self.model.to(parameter.device)

        self.model_teacher.load_state_dict(weights)
        self.model_teacher = self.model_teacher.to(parameter.device)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=parameter.learning_rate,
            fused=True if parameter.device == "cuda" else False,
        )


def cross_entropy_dirichlet(prediction: torch.Tensor, target: torch.Tensor):
    alpha = prediction + 1
    diriclet_strength = alpha.sum(dim=1, keepdim=True)
    return (target * (torch.digamma(diriclet_strength) - torch.digamma(alpha))).sum(1)


def KL_divergence_dirichlet(prediction: torch.Tensor, target: torch.Tensor):
    alpha = prediction + 1
    n_class = torch.tensor(prediction.size(1))
    approx_alpha = target + (1 - target) * alpha

    first_term = torch.lgamma(approx_alpha.sum(dim=1))
    first_term -= torch.lgamma(n_class) + torch.lgamma(approx_alpha).sum(dim=1)
    second_term = (
        (approx_alpha - 1)
        * (
            torch.digamma(approx_alpha)
            - torch.digamma(approx_alpha.sum(dim=1, keepdim=True))
        )
    ).sum(dim=1)
    return first_term + second_term


def overall_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_t: torch.Tensor,
):
    prediction = prediction.relu()
    loss = cross_entropy_dirichlet(prediction, target)
    loss += lambda_t * KL_divergence_dirichlet(
        prediction,
        target,
    )
    return loss.mean()


def consistency_loss(prediction: torch.Tensor, target: torch.Tensor):
    return (prediction - target).pow(2).mean()


def overall_with_uncertainty(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lambda_t: torch.Tensor,
    uncertainty: torch.Tensor = None,
):
    prediction = prediction.relu()
    loss = cross_entropy_dirichlet(prediction, target)
    loss += lambda_t * KL_divergence_dirichlet(
        prediction,
        target,
    )
    if uncertainty is None:
        uncertainty = torch.ones_like(loss)
    return ((1 - uncertainty) * loss).mean()


def create_dataloader(
    path: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    global_dataset = HeDataset(directory_path=path)
    SPLIT_PERCENTAGE = 0.8

    train_dataset, test_dataset = random_split(
        global_dataset,
        [SPLIT_PERCENTAGE, 1 - SPLIT_PERCENTAGE],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=num_workers,
    )
    return train_dataloader, test_dataloader
