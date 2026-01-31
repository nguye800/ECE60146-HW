"""
ECE 60146 - Homework 2
Name: Lucas Nguyen
Email: nguye800@purdue.edu

ImageNet Data Loading and Augmentation
"""

import os
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms


# ============================================================
# REQUIRED CLASSES
# Labels match both the provided subset AND Hugging Face ImageNet
# ============================================================

REQUIRED_CLASSES = {
    1: "goldfish",
    151: "Chihuahua",
    281: "tabby cat",
    291: "lion",
    325: "sulphur butterfly",
    386: "African elephant",
    430: "basketball",
    466: "bullet train",
    496: "Christmas stocking",
    950: "orange",
}


def get_class_name(label):
    """Get class name from integer label."""
    return REQUIRED_CLASSES.get(label, f"class_{label}")


# ============================================================
# TRANSFORMS
# ============================================================

transform_custom = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(3, (0.3, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_basic = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# OPTION A: Dataset class for provided ImageNet subset
# ============================================================

class ImageNetSubset(Dataset):
    """
    Dataset for loading the provided ImageNet subset.

    Folder structure:
        imagenet_subset/
            1/  (goldfish)
                00001.JPEG
                ...
            15/ (robin)
                ...

    Args:
        root (str): Path to imagenet_subset folder
        class_labels (list): List of integer labels to load
        images_per_class (int): Number of images per class
        transform (callable): Transform to apply to images
    """

    def __init__(self, root, class_labels, images_per_class=5, transform=None):
        self.root = root
        self.class_labels = class_labels
        self.images_per_class = images_per_class
        self.transform = transform

        self.samples = []  # List of (image_path, label)
        self._load_samples()

        print(f"Loaded {len(self.samples)} images from {len(class_labels)} classes")

    def _load_samples(self):
        """Load image paths for each requested class."""
        for label in self.class_labels:
            base_path = os.path.join(self.root, str(label))
            count = 0
            for img in sorted(os.listdir(base_path)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(base_path, img), label))
                    count += 1

                if count == self.images_per_class:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            image (Tensor): Transformed image
            label (int): Class label
        """
        img_path = self.samples[index][0]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.samples[index][1]

# ============================================================
# Custom Dataset for your own images
# ============================================================

class CustomDataset(Dataset):
    """Dataset for loading custom images from a folder."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.image_paths = [os.path.join(root, img) for img in os.listdir(root) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        print(f"Found {len(self.image_paths)} images in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, 0


# ============================================================
# Utility Functions
# ============================================================

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def show_images(images, titles, rows, cols, figsize=(15, 10), save_path=None):
    """Display a grid of images."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        if i >= len(axes):
            break

        if isinstance(img, torch.Tensor):
            if img.min() < 0 or img.max() > 1:
                img = denormalize(img)
            img = img.permute(1, 2, 0).numpy()

        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def set_seed(seed=60146):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    required_labels = list(REQUIRED_CLASSES.keys())
    print("Required classes:", required_labels)
    for label in required_labels:
        print(f"{label}: {get_class_name(label)}")

    # Task 1: Load ImageNet (50 images: 10 classes x 5 images)
    print("\n" + "=" * 60)
    print("Task 1: Loading ImageNet")
    print("=" * 60)

    imagenet = ImageNetSubset(root='D:\Lucas College\Purdue\Y4\ECE60146-HW\HW2\imagenet_subset',
                              class_labels=required_labels,
                              images_per_class=5,
                              transform=transform_basic)
    
    dataloader = DataLoader(
        imagenet,
        batch_size=5,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    it = iter(dataloader)
    print("Total Images:", len(imagenet))
    for i in range(10):
        images, labels = next(it)
        print("Class Label:", labels)

    # Task 2: Visualize ImageNet (1 image per class = 10 images
    print("\n" + "=" * 60)
    print("Task 2: Visualizing ImageNet")
    print("=" * 60)

    imagenet = ImageNetSubset(root='D:\Lucas College\Purdue\Y4\ECE60146-HW\HW2\imagenet_subset',
                              class_labels=required_labels,
                              images_per_class=1,
                              transform=transform_basic)
    
    dataloader = DataLoader(
        imagenet,
        batch_size=10,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(2, 5, figsize=(12, 4))

    for i in range(10):
        img = images[i].permute(1, 2, 0)  # CHW → HWC
        axes[i//5][i%5].imshow(img)
        axes[i//5][i%5].axis("off")
        axes[i//5][i%5].set_title(f"Image {labels[i]}")

    plt.tight_layout()
    plt.savefig("imagenet_samples.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Task 3: Show augmentation effects
    print("\n" + "=" * 60)
    print("Task 3: Augmentation comparison")
    print("=" * 60)

    imagenet_orig = ImageNetSubset(root='D:\Lucas College\Purdue\Y4\ECE60146-HW\HW2\imagenet_subset',
                              class_labels=required_labels,
                              images_per_class=1,
                              transform=transform_basic)
    imagenet_aug = ImageNetSubset(root='D:\Lucas College\Purdue\Y4\ECE60146-HW\HW2\imagenet_subset',
                              class_labels=required_labels,
                              images_per_class=1,
                              transform=transform_custom)
    
    dataloader_orig = DataLoader(
        imagenet_orig,
        batch_size=3,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    dataloader_aug = DataLoader(
        imagenet_aug,
        batch_size=3,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    images_orig, labels_orig = next(iter(dataloader_orig))
    images_aug, labels_aug = next(iter(dataloader_aug))
    fig, axes = plt.subplots(2, 3, figsize=(12, 14))

    for i in range(3):
        img_orig = images_orig[i].permute(1, 2, 0)  # CHW → HWC
        axes[0][i].imshow(img_orig)
        axes[0][i].axis("off")
        axes[0][i].set_title(f"Image {labels_orig[i]}")

        img_aug = images_aug[i].permute(1, 2, 0)  # CHW → HWC
        axes[1][i].imshow(img_aug)
        axes[1][i].axis("off")
        axes[1][i].set_title(f"Image Aug {labels_aug[i]}")

    plt.tight_layout()
    plt.savefig("imagenet_aug.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Task 4: Custom dataset
    print("\n" + "=" * 60)
    print("Task 4: Custom dataset")
    print("=" * 60)

    fig, axes = plt.subplots(2, 5, figsize=(12, 4))

    customDataset_1 = CustomDataset(root='D:/Lucas College/Purdue/Y4/ECE60146-HW/HW2/imagenet_subset/151', transform=transform_custom)
    customDataset_2 = CustomDataset(root='D:/Lucas College/Purdue/Y4/ECE60146-HW/HW2/imagenet_subset/281', transform=transform_custom)

    dataset20 = ConcatDataset([customDataset_1, customDataset_2])
    
    dataloader_custom = DataLoader(
        dataset20,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    num_generated = 0
    for epoch in range(5):
        images, labels = next(iter(dataloader_custom))
        for i in range(10):
            num_generated += 1
            if epoch == 4:
                img = images[i].permute(1, 2, 0)  # CHW → HWC
                axes[i//5][i%5].imshow(img)
                axes[i//5][i%5].axis("off")
                axes[i//5][i%5].set_title(f"Image {labels[i]}")

    print("Number of Augmented Images Generated:", num_generated)
    plt.tight_layout()
    plt.savefig("custom_samples.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Task 5: DataLoader performance
    print("\n" + "=" * 60)
    print("Task 5: DataLoader performance")
    print("=" * 60)

    imagenet = ImageNetSubset(root='D:\Lucas College\Purdue\Y4\ECE60146-HW\HW2\imagenet_subset',
                              class_labels=required_labels,
                              images_per_class=10,
                              transform=transform_custom)
    
    for pair in [(16 ,0), (16 ,4), (64 ,0), (64 ,4)]:
        t0 = time.perf_counter()
        dataloader = DataLoader(
            imagenet,
            batch_size=pair[0],
            shuffle=False,
            num_workers=pair[1],
            pin_memory=True
        )
        seen = 0
        for images, labels in dataloader:
            seen += images.size(0)
        t1 = time.perf_counter()
        print(f"For batch size: {pair[0]} and num workers: {pair[1]} the time was {t1-t0}")

    # Task 6: RGB statistics
    print("\n" + "=" * 60)
    print("Task 6: RGB statistics")
    print("=" * 60)

    def channel_min_max(x):
        return [(x[c].min().item(), x[c].max().item()) for c in range(3)]

    img = Image.open('D:/Lucas College/Purdue/Y4/ECE60146-HW/HW2/imagenet_subset/1/00001.JPEG').convert("RGB")
    img_pre = transform_basic(img)
    img_post = transform_custom(img)

    img_stats = channel_min_max(np.array(img)) 
    pre_stats = channel_min_max(img_pre)
    post_stats = channel_min_max(img_post)

    print("Before all transformation")
    print("R:", img_stats[0], "G:", img_stats[1], "B:", img_stats[2])

    print("After ToTensor before Normalize")
    print("R:", pre_stats[0], "G:", pre_stats[1], "B:", pre_stats[2])

    print("After Normalize")
    print("R:", post_stats[0], "G:", post_stats[1], "B:", post_stats[2])

    # Task 7: Reproducibility
    print("\n" + "=" * 60)
    print("Task 7: Reproducibility")
    print("=" * 60)

    def set_seed(seed = 60146):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed()

    imagenet = ImageNetSubset(root='D:\Lucas College\Purdue\Y4\ECE60146-HW\HW2\imagenet_subset',
                              class_labels=required_labels,
                              images_per_class=10,
                              transform=transform_custom)
    dataloader = DataLoader(
            imagenet,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(2):
        img = images[i].permute(1, 2, 0)  # CHW → HWC
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Image {labels[i]}")

    plt.tight_layout()
    plt.savefig("imagenet_samples.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
