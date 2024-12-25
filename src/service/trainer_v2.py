import math
import time

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src.experiment.he_experiment import HeExperiment, overall_loss
from src.loss import dice_index
from src.service.base_trainer import BaseTrainer
from src.service.ema import EMA
from src.service.parameter import Parameter


class Trainer(BaseTrainer):
    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
        torch.manual_seed(99951499)

        experiment = HeExperiment(parameter=self.parameter)
        self.train_dataloader = experiment["train_dataloader"]
        self.test_dataloader = experiment["test_dataloader"]
        self.model = experiment["model"]
        self.scheduler = experiment["scheduler"]
        self.preprocessor = experiment["preprocessor"]
        self.optimizer = experiment["optimizer"]
        self.loss_fn = overall_loss
        self.ema = EMA(self.model, decay=0.999)

    def train(self):
        if torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.bfloat16

        for epoch in range(self.parameter.epoch):
            print(f"Training epoch {epoch + 1}, ", end="")

            # Unfreeze backbone at epoch 2
            if epoch == 2:
                for parameter in self.model.backbone.parameters():
                    parameter.requires_grad = True

            initial_time = time.time()
            self._train_one_epoch(
                epoch=epoch,
                model=self.model,
                dataloader=self.dataloader_train,
                optimizer=self.optimizer,
                loss_fn=self.loss_fn,
                preprocess=self.preprocess,
                device=self.device,
                dtype=dtype,
                scheduler=self.scheduler,
            )
            time_taken = time.time() - initial_time
            print(f"time_taken: {time_taken}s")

            if self.dataloader_test is not None:
                self._eval_one_epoch(
                    epoch=epoch,
                    model=self.model,
                    dataloader=self.dataloader_test,
                    loss_fn=self.loss_fn,
                    preprocess=self.preprocess,
                    device=self.device,
                    train_dataset_length=len(self.dataloader_train),
                    dtype=dtype,
                )
            self._save(model=self.model, epoch=epoch)

    def _train_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        preprocess: v2.Compose,
        device: str,
        dtype,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        rate_to_print = max(
            math.floor(len(dataloader) * self.parameter.train_report_rate), 1
        )
        running_loss = 0.0
        running_iou = 0.0

        lambda_t = torch.tensor((epoch / 100) * 0.02)
        scaler = torch.cuda.amp.grad_scaler.GradScaler()
        for index, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.autocast(device_type=device, dtype=dtype):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = data

                inputs, labels = inputs.to(device), labels.to(device)
                inputs, labels = preprocess(inputs, labels)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels, lambda_t)
                alpha = outputs.relu() + 1
                diriclet_strength = alpha.sum(dim=1, keepdim=True)
                prediction = alpha / diriclet_strength
                iou_score = dice_index(prediction, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer)
            if scheduler is not None:
                scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_iou += iou_score.item()

            if index % rate_to_print == (rate_to_print - 1):
                current_training_sample = epoch * len(dataloader) + index + 1
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

    def _eval_one_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        dataloader: DataLoader,
        preprocess: v2.Compose,
        loss_fn,
        device: str,
        train_dataset_length: int,
        dtype,
    ):
        sum_loss = 0.0
        sum_iou = 0.0
        lambda_t = torch.tensor((epoch / 100) * 0.02)

        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=dtype):
                for data in dataloader:
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels = data

                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels = preprocess(inputs, labels)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels, lambda_t)
                    alpha = outputs.relu() + 1
                    diriclet_strength = alpha.sum(dim=1, keepdim=True)
                    prediction = alpha / diriclet_strength
                    iou_score = dice_index(prediction, labels)

                    sum_loss += loss.item()
                    sum_iou += iou_score.item()

        iteration = (epoch + 1) * train_dataset_length
        avg_loss = sum_loss / len(dataloader)
        avg_iou = sum_iou / len(dataloader)
        self.writer_test.add_scalar("loss", avg_loss, iteration)
        self.writer_test.add_scalar("iou_score", avg_iou, iteration)
