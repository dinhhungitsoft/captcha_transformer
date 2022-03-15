import albumentations
import torch
import numpy as np
from PIL import Image, ImageFile
import config
import glob
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CaptchaDataset:
    def __init__(self, image_paths, targets, test_orig_targets=None,resize=None):
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.raw_targets = test_orig_targets
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(always_apply=True)
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        raw_targets = []
        if self.raw_targets is not None:
            raw_targets = self.raw_targets[item]

        if self.resize is not None:
            # self.resize: (h, w)
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2,0,1)).astype(np.float32)
        if self.raw_targets is not None:
            return {
                "images": torch.tensor(image, dtype=torch.float32),
                "targets": torch.tensor(targets, dtype=torch.long),
                "raw_targets": raw_targets
            }
        else:
            return {
                "images": torch.tensor(image, dtype=torch.float32),
                "targets": torch.tensor(targets, dtype=torch.long)                
            }

if __name__ == "__main__":
    ds = CaptchaDataset(glob.glob(os.path.join("*.png")))