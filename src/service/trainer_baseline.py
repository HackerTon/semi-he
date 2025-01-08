import math
from typing import Tuple

import torch
import torch.nn.functional as fn
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src.dataloader.dataset.he_dataset_direct import HeDataset
from src.dataloader.dataset.ocelot_dataset import OcelotDataset
from src.dataloader.transform import ImagenetNormalize, ToNormalized
from src.loss import dice_index
from src.model.model import UNETNetwork
from src.service.base_trainer import BaseTrainer
from src.service.parameter import Parameter


class TrainerBaseline(BaseTrainer):
    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
        self._setup()

    def _setup(self):
        torch.manual_seed(99951499)
        self.train_dataloader, self.test_dataloader = create_dataloader(
            path=self.parameter.data_path,
            batch_size=self.parameter.batch_size_train,
        )
        self.model = UNETNetwork(number_class=3)
        weights = torch.load(self.parameter.pretrain_path)
        self.model.load_state_dict(weights, strict=False)
        self.preprocessor = v2.Compose(
            [
                ToNormalized(),
                ImagenetNormalize(),
            ]
        )
        self.model = self.model.to(self.parameter.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.parameter.learning_rate,
            fused=True if self.parameter.device == "cuda" else False,
        )
        self.loss_fn = baseline_loss
        self.dtype = torch.float16

    def train(self):
        for epoch in tqdm(range(self.parameter.epoch)):
            if epoch == 10:
                for parameter in self.model.parameters():
                    parameter.requires_grad = True
            self._train_one_epoch(epoch)
            self._eval_one_epoch(epoch)
            self._save(model=self.model, epoch=epoch)

    def _train_one_epoch(self, epoch):
        rate_to_print = max(
            math.floor(len(self.train_dataloader) * self.parameter.train_report_rate), 1
        )
        running_loss = 0.0
        running_iou = 0.0
        self.model.train()
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for index, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.parameter.device, dtype=self.dtype):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs, labels = (
                    inputs.to(self.parameter.device),
                    labels.to(self.parameter.device),
                )
                inputs, labels = self.preprocessor(inputs, labels)

                output_logits = self.model(inputs)
                loss = self.loss_fn(output_logits, labels)
                iou_score = dice_index(output_logits.sigmoid(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer=self.optimizer)
            scaler.update()

            running_loss += loss.item()
            running_iou += iou_score.item()

            if index % rate_to_print == (rate_to_print - 1):
                current_training_sample = epoch * len(self.train_dataloader) + index + 1
                self.writer_train.add_scalar(
                    "loss",
                    running_loss / rate_to_print,
                    current_training_sample,
                )
                self.writer_train.add_scalar(
                    "iou_score",
                    running_iou / rate_to_print,
                    current_training_sample,
                )
                running_loss = 0.0
                running_iou = 0.0

    def _eval_one_epoch(self, epoch):
        sum_loss = 0.0
        sum_iou = 0.0

        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.parameter.device, dtype=self.dtype):
                for data in self.test_dataloader:
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels = data

                    inputs, labels = (
                        inputs.to(self.parameter.device),
                        labels.to(self.parameter.device),
                    )
                    inputs, labels = self.preprocessor(inputs, labels)
                    output_logits = self.model(inputs)
                    loss = self.loss_fn(output_logits, labels)
                    iou_score = dice_index(output_logits.sigmoid(), labels)

                sum_loss += loss.item()
                sum_iou += iou_score.item()

        iteration = (epoch + 1) * len(self.train_dataloader)
        avg_loss = sum_loss / len(self.test_dataloader)
        avg_iou = sum_iou / len(self.test_dataloader)
        self.writer_test.add_scalar("loss", avg_loss, iteration)
        self.writer_test.add_scalar("iou_score", avg_iou, iteration)


def baseline_loss(pred: torch.Tensor, target: torch.Tensor):
    crossentropy_loss = fn.binary_cross_entropy_with_logits(pred, target)
    dice_loss = 1 - dice_index(pred.sigmoid(), target)
    return crossentropy_loss + dice_loss


def create_dataloader(
    path: str,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    global_dataset = OcelotDataset(directory_path=path)
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
