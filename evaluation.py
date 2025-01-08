import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.io.image import write_png
from torchvision.transforms.v2 import Compose
from tqdm import tqdm

from src.dataloader.dataset.he_dataset import HeDataset
from src.dataloader.dataset.he_dataset_direct import HeDatasetBaseline
from src.dataloader.transform import ImagenetNormalize, ToNormalized
from src.model.model import UNETNetwork, UNETNetworkModi

BATCH_SIZE = 128


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


def accuracy_per_channel(prediction, labels):
    correct_per_class = ((labels == 1) == (prediction > 0.5)).sum(dim=[0, 2, 3])
    return correct_per_class / prediction[:, 0, ...].flatten().size(0)


def dice_index_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon=1e-4,
):
    pred_flat = pred.permute([1, 0, 2, 3]).flatten(1)
    label_flat = target.permute([1, 0, 2, 3]).flatten(1)
    nominator = 2 * torch.sum(pred_flat * label_flat, dim=1)
    denominator = torch.sum(pred_flat, dim=1) + torch.sum(label_flat, dim=1)
    return (nominator + epsilon) / (denominator + epsilon)


def write_sample(image, label, directory: Path, index):
    image_path = directory.joinpath(f"{index}_image.png")
    label_path = directory.joinpath(f"{index}_label.png")
    write_png(image, str(image_path))
    write_png(label, str(label_path))


def baseline(model, dataloader, preprocessor):
    iou_array = torch.zeros([len(dataloader.dataset) // BATCH_SIZE + 1, 3])
    accuracy_array = torch.zeros([len(dataloader.dataset) // BATCH_SIZE + 1, 3])

    for i, (image, mask) in tqdm(enumerate(dataloader)):
        image_after, mask = preprocessor(image.cuda(), mask.cuda())
        with torch.no_grad():
            prediction = model(image_after)
        prediction = prediction.sigmoid()

        iou_array[i] = dice_index_per_channel(prediction, mask)
        accuracy_array[i] = accuracy_per_channel(prediction, mask)

    print(f"IOU: {iou_array.mean(dim=0)}")
    print(f"ACC: {accuracy_array.mean(dim=0)}")


def dirichlet(model, dataloader, preprocessor):
    iou_array = torch.zeros([len(dataloader.dataset) // BATCH_SIZE + 1, 3])
    accuracy_array = torch.zeros([len(dataloader.dataset) // BATCH_SIZE + 1, 3])

    for i, (image, mask) in tqdm(enumerate(dataloader)):
        image_after, mask = preprocessor(image.cuda(), mask.cuda())
        with torch.no_grad():
            prediction = model(image_after)

        output_belief, output_uncertainty = combined_dirichlet(
            prediction[0].relu(),
            prediction[1].relu(),
        )
        prediction = convert_belief_mass_to_prediction(
            output_belief,
            output_uncertainty,
        )

        iou_array[i] = dice_index_per_channel(prediction, mask)
        accuracy_array[i] = accuracy_per_channel(prediction, mask)

    print(f"IOU: {iou_array.mean(dim=0)}")
    print(f"ACC: {accuracy_array.mean(dim=0)}")


def run(namespace):
    torch.manual_seed(99951499)
    if namespace.mode == "baseline":
        model = UNETNetwork(number_class=3)
    elif namespace.mode == "dirichlet":
        model = UNETNetworkModi(numberClass=3)
    else:
        print("mode is not baseline or dirichlet")

    model.load_state_dict(torch.load(namespace.model))
    model.eval()
    model.cuda()
    preprocessor = Compose([ToNormalized(), ImagenetNormalize()])

    dataset = HeDatasetBaseline(namespace.dataset_dir)
    train_dataset, test_dataset = random_split(
        dataset,
        [0.8, 0.2],
    )

    dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
    )

    if namespace.mode == "baseline":
        baseline(model, dataloader, preprocessor)
    elif namespace.mode == "dirichlet":
        dirichlet(model, dataloader, preprocessor)
    else:
        print("mode is not baseline or dirichlet")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model path with has pt")
    parser.add_argument("--dataset_dir", required=True, help="he dataset")
    parser.add_argument("--mode", required=True, help="baseline or dirichlet")
    parsed_data = parser.parse_args()
    run(parsed_data)
