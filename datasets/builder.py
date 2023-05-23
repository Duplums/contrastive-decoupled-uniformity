import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import Type, Callable, List
from PIL import Image
import torch
import os
import numpy as np
# custom PyTorch implementation of public datasets
from datasets.imagenet100 import ImageNet100
from datasets.chexpert import CheXpert
from datasets.cub import CUB
from datasets.utzappos import UTZappos
# torchvision implementation of benchmarking datasets
from torchvision.datasets import CIFAR10, CIFAR100
global DatasetWithPrior

class NTransform:
    """Creates a pipeline that applies a transformation pipeline multiple times."""

    def __init__(self, base_transform: Callable, n_views: int=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x: Image) -> torch.Tensor:
        return torch.stack([self.base_transform(x) for _ in range(self.n_views)])

class ColorDistortion:
    def __init__(self, s: float=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        self.color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])

    def __call__(self, x):
        return self.color_distort(x)

def build_transform_pipeline(args):
    mean_std = {
        "imagenet100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "cub200": ((0.4863, 0.4999, 0.4312), (0.2070, 0.2018, 0.2428)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        "utzappos": ((0.8342, 0.8142, 0.8081), (0.2804, 0.3014, 0.3072)),
        "chexpert": (0.5024, 0.2898)
    }
    tf = None
    img_size = 224
    if args.db in ["cifar10", "cifar100", "utzappos"]:
        img_size = 32

    if args.db in ["imagenet100", "cub200", "cifar10", "cifar100", "utzappos"]:
        tf = NTransform(transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1)),
            transforms.RandomHorizontalFlip(),
            ColorDistortion(s=1.0),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std[args.db])
        ]), args.n_views)
    elif args.db == "chexpert":
        # These transforms follow "Big Self-Supervised Models Advance Medical Image Classification", ICCV 2021
        tf = NTransform(transforms.Compose([
            transforms.RandomResizedCrop(img_size, interpolation=Image.BICUBIC),
            transforms.RandomRotation(45),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std[args.db])
        ]), args.n_views)
    else:
        raise ValueError("Unknown dataset: %s"%args.db)
    return tf

def dataset_with_prior(DatasetClass: Type[Dataset], prior_path: str) -> Type[Dataset]:
    """ Factory dataset that returns (sample, prior) instead of
        (sample, label) with target class `label`.
    Args:
        DatasetClass: Dataset class to be wrapped
        prior_path: str, path to prior.npz
    Returns:
        Dataset returning prior as labels
    """
    class DatasetWithPrior(DatasetClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not os.path.isfile(prior_path):
                raise FileNotFoundError("Check %s"%prior_path)
            try:
                pth_loaded = np.load(prior_path)
                self.prior = pth_loaded["prior"]
                true_labels = pth_loaded["labels"]
            except Exception:
                raise ValueError("Check numpy array: %s"%prior_path)
            if len(self.prior.shape) == 3:
                self.prior = self.prior[:, 0]
            if len(true_labels.shape) == 3:
                true_labels = true_labels[:, 0]
            assert len(self.prior) == len(self), "Inconsistent # of samples"
            if hasattr(self, "targets"):
                assert np.all(true_labels.squeeze() == self.targets), "Inconsistent labels"
            print("Weak labels loaded.", flush=True)

        def __getitem__(self, idx):
            sample, _ = super().__getitem__(idx)
            target = self.prior[idx]
            return sample, target
    return DatasetWithPrior


def prepare_dataset(args) -> Dataset:
    transform = build_transform_pipeline(args)

    dataset2cls = dict(imagenet100=ImageNet100,
                       cifar10=CIFAR10,
                       cifar100=CIFAR100,
                       cub200=CUB,
                       utzappos=UTZappos,
                       chexpert=CheXpert)
    extra_kwargs = {}

    if args.db in ["cifar10", "cifar100", "cub200", "utzappos"]:
        extra_kwargs["download"] = True

    if args.weaklabels:
        prior_path = dataset2cls[args.db].prior_path
        dataset_cls = dataset_with_prior(dataset2cls[args.db], prior_path)
    else:
        dataset_cls = dataset2cls[args.db]

    return dataset_cls(args.root, transform=transform, **extra_kwargs)



