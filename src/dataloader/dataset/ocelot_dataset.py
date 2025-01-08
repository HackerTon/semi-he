from itertools import chain
from pathlib import Path
from shutil import rmtree

import torch
from torch.utils.data import Dataset
from torchvision.io.image import ImageReadMode, read_image, write_png
from torchvision.transforms.functional import InterpolationMode, resize


class OcelotDataset(Dataset):
    def __init__(self, directory_path: str):
        self.directory = Path(directory_path)
        self.cache_directory = Path(directory_path).joinpath(".tmp")
        self._initialize_dataset()

    def _initialize_dataset(self):
        train_images = list(self.directory.glob("images/train/tissue/*.jpg"))
        total_num_samples = 4 * 4 * (len(train_images))
        # test_images = list(self.directory.glob("imagesTs/*.png"))
        # total_num_samples = 4 * 4 * (len(train_images) + len(test_images))

        # Check the number of images in cache
        # If complete, return
        if self.cache_directory.exists():
            num_images = len(list(self.cache_directory.glob("*image.png")))
            num_labels = len(list(self.cache_directory.glob("*label.png")))

            if num_images == num_labels == total_num_samples:
                self.dataset_length = total_num_samples
                return

        self.cache_directory.mkdir(exist_ok=True)
        rmtree(str(self.cache_directory))
        self.cache_directory.mkdir(exist_ok=True)
        self._cache_dataset(chain(train_images))
        self.dataset_length = total_num_samples

    def _cache_dataset(self, images):
        count = 0
        for image_path in images:
            # label_path = "_".join(str(image_path).split("_")[:-1]) + ".png"
            # label_path = label_path.replace("imagesTr", "labelsTr")
            # label_path = label_path.replace("imagesTs", "labelsTs")
            label_path = str(image_path).replace("images", "annotations")
            label_path = label_path.replace("jpg", "png")

            image = read_image(str(image_path), ImageReadMode.RGB)
            label = read_image(label_path, ImageReadMode.GRAY)

            # resized_image = resize(image, [1024, 1024], InterpolationMode.NEAREST)
            # resized_label = resize(label, [1024, 1024], InterpolationMode.NEAREST)

            for i in range(1, 5):
                for j in range(1, 5):
                    y_slice = slice((i - 1) * 256, i * 256)
                    x_slice = slice((j - 1) * 256, j * 256)

                    block_image = image[..., y_slice, x_slice]
                    block_label = label[..., y_slice, x_slice]

                    write_png(
                        block_image,
                        str(self.cache_directory.joinpath(f"{count}_image.png")),
                    )
                    write_png(
                        block_label,
                        str(self.cache_directory.joinpath(f"{count}_label.png")),
                    )
                    count += 1

    def __len__(self):
        return self.dataset_length

    def get_mask(self, label):
        return (
            torch.concatenate(
                [
                    (label == 1),  # BACKGROUND class
                    (label == 2),  # CANCER class
                    (label == 255),  # UNKNOWN class
                ]
            )
            * 255
        )

    def __getitem__(self, index):
        image_path = str(self.cache_directory.joinpath(f"{index}_image.png"))
        label_path = str(self.cache_directory.joinpath(f"{index}_label.png"))
        image = read_image(image_path, ImageReadMode.RGB)
        label = read_image(label_path, ImageReadMode.GRAY)
        label = self.get_mask(label).to(torch.uint8)
        return image, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.io import write_jpeg

    dataset = OcelotDataset("/mnt/storage/ocelot2023_v1.0.1")
    i = 0
    for image, mask in DataLoader(dataset, shuffle=False, num_workers=4):
        i += 1
    print(i)
