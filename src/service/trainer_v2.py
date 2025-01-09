import math

import torch
from tqdm import tqdm

from src.experiment.he_experiment import (
    HeExperiment,
    consistency_loss,
    overall_with_uncertainty,
)
from src.loss import dice_index
from src.service.base_trainer import BaseTrainer
from src.service.ema import EMAFunction
from src.service.parameter import Parameter


class TrainerV2(BaseTrainer):
    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
        torch.manual_seed(99951499)

        experiment = HeExperiment(parameter=self.parameter)
        self.train_dataloader = experiment.train_dataloader
        self.test_dataloader = experiment.test_dataloader
        self.model = experiment.model
        self.model_teacher = experiment.model_teacher
        self.scheduler = experiment.scheduler
        self.preprocessor = experiment.preprocessor
        self.optimizer = experiment.optimizer
        self.loss_fn = overall_with_uncertainty
        self.ema_fn = EMAFunction()
        self.dtype = torch.float16

    def train(self):
        for epoch in tqdm(range(self.parameter.epoch)):
            # Unfreeze backbone at epoch 2
            # if epoch == 10:
            #     for parameter in self.model.parameters():
            #         parameter.requires_grad = True
            self._train_one_epoch(epoch)
            if self.test_dataloader is not None:
                self._eval_one_epoch(epoch)
            self._save(model=self.model_teacher, epoch=epoch)

    def _train_one_epoch(self, epoch):
        times_to_update_ema = 10
        rate_to_print = max(
            math.floor(len(self.train_dataloader) * self.parameter.train_report_rate), 1
        )
        running_loss = 0.0
        running_iou = 0.0
        self.model.train()
        lambda_t = torch.tensor((epoch / self.parameter.epoch) * 0.02)
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        for index, data in enumerate(self.train_dataloader):
            with torch.autocast(device_type=self.parameter.device, dtype=self.dtype):
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels, uncertainty = data

                inputs, labels, uncertainty = (
                    inputs.to(self.parameter.device),
                    labels.to(self.parameter.device),
                    uncertainty.to(self.parameter.device),
                )
                inputs, labels = self.preprocessor(inputs, labels)

                outputs = self.model(inputs)
                output_belief, output_uncertainty = combined_dirichlet(
                    outputs[0].relu(),
                    outputs[1].relu(),
                )
                evidences = output_belief * 3 / output_uncertainty
                prediction = convert_belief_mass_to_prediction(
                    output_belief,
                    output_uncertainty,
                )
                loss = self.loss_fn(
                    evidences,
                    labels,
                    lambda_t,
                    uncertainty,
                )

                with torch.no_grad():
                    teacher_output = self.model_teacher(inputs)
                    teacher_output_belief, teacher_output_uncertainty = (
                        combined_dirichlet(
                            teacher_output[0].relu(),
                            teacher_output[1].relu(),
                        )
                    )
                    teacher_prediction = convert_belief_mass_to_prediction(
                        teacher_output_belief,
                        teacher_output_uncertainty,
                    )

                loss += consistency_loss(prediction, teacher_prediction)
                iou_score = dice_index(prediction, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Update teacher model
            if (index + 1) % times_to_update_ema == 0:
                self.ema_fn.update_parameter_to_model(
                    0.999,
                    self.model_teacher,
                    self.model.named_parameters(),
                )

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
        lambda_t = torch.tensor((epoch / self.parameter.epoch) * 0.02)

        self.model.eval()
        self.model_teacher.eval()

        for data in self.test_dataloader:
            with torch.autocast(device_type=self.parameter.device, dtype=self.dtype):
                with torch.no_grad():
                    inputs: torch.Tensor
                    labels: torch.Tensor
                    inputs, labels, uncertainty = data

                    inputs, labels, uncertainty = (
                        inputs.to(self.parameter.device),
                        labels.to(self.parameter.device),
                        uncertainty.to(self.parameter.device),
                    )
                    inputs, labels = self.preprocessor(inputs, labels)

                    outputs = self.model_teacher(inputs)
                    output_belief, output_uncertainty = combined_dirichlet(
                        outputs[0].relu(),
                        outputs[1].relu(),
                    )
                    evidences = output_belief * 3 / output_uncertainty
                    prediction = convert_belief_mass_to_prediction(
                        output_belief,
                        output_uncertainty,
                    )
                    loss = self.loss_fn(evidences, labels, lambda_t, uncertainty)
                    iou_score = dice_index(prediction, labels)

                    sum_loss += loss.item()
                    sum_iou += iou_score.item()

        iteration = (epoch + 1) * len(self.train_dataloader)
        avg_loss = sum_loss / len(self.test_dataloader)
        avg_iou = sum_iou / len(self.test_dataloader)
        self.writer_test.add_scalar("loss", avg_loss, iteration)
        self.writer_test.add_scalar("iou_score", avg_iou, iteration)


def combined_dirichlet(evidence_a, evidence_b, k_class=3):
    """
    evidences: [batchsize, channel, H, W]
    """

    alpha_a = evidence_a + 1
    alpha_b = evidence_b + 1
    dirichlet_strength_a = alpha_a.sum(dim=1, keepdim=True)
    dirichlet_strength_b = alpha_b.sum(dim=1, keepdim=True)
    belief_a = evidence_a / dirichlet_strength_a
    belief_b = evidence_b / dirichlet_strength_b
    uncertainty_a = k_class / dirichlet_strength_a
    uncertainty_b = k_class / dirichlet_strength_b

    sum_conflicts = (
        belief_a.unsqueeze(2)
        * belief_b.unsqueeze(1)
        * (1 - torch.eye(k_class, device="cuda").view(1, k_class, k_class, 1, 1))
    ).sum([1, 2])

    scale_factor = 1 / (1 - sum_conflicts).unsqueeze(1)

    combined_beliefs = (
        scale_factor * belief_a * belief_b
        + belief_a * uncertainty_b
        + belief_b * uncertainty_a
    )
    combined_uncertainties = scale_factor * uncertainty_a * uncertainty_b
    return combined_beliefs, combined_uncertainties


def convert_belief_mass_to_prediction(belief, uncertainty, k_class=3):
    return belief + uncertainty / k_class
