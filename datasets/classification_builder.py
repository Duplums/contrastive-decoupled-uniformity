import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets.builder import NTransform, ColorDistortion
# custom PyTorch implementation of public datasets
from datasets.imagenet100 import ImageNet100
from datasets.chexpert import CheXpert
from datasets.cub import CUB
from datasets.utzappos import UTZappos
# torchvision implementation of benchmarking datasets
from torchvision.datasets import CIFAR10, CIFAR100


def build_transform_pipeline(args):
    mean_std = {
        "imagenet100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        "cub200": ((0.4863, 0.4999, 0.4312), (0.2070, 0.2018, 0.2428)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2009, 0.1984, 0.2023)),
        "utzappos": ((0.8342, 0.8142, 0.8081), (0.2804, 0.3014, 0.3072)),
        "chexpert": (0.5024, 0.2898)
    }
    tf = dict(train=None, val=None)
    img_size = 224
    if args.db in ["cifar10", "cifar100", "utzappos"]:
        img_size = 32

    if args.db in ["cifar10", "cifar100", "utzappos"]:
        tf["val"] = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std[args.db])
        ])
    elif args.db in ["cub200", "imagenet100"]:
        tf["val"] = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std[args.db]),
        ])
    elif args.db == "chexpert":
        tf["val"] = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std[args.db]),
        ])
    else:
        raise ValueError("Unknown dataset: %s"%args.db)

    if args.db == "imagenet100":
        tf["train"] = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            ColorDistortion(s=1),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std[args.db])
        ])
    else:
        tf["train"] = tf["val"]
    return tf



def prepare_datasets(args) -> [Dataset, Dataset]:

    transform = build_transform_pipeline(args)

    dataset2cls = dict(imagenet100=ImageNet100,
                       cifar10=CIFAR10,
                       cifar100=CIFAR100,
                       cub200=CUB,
                       utzappos=UTZappos,
                       chexpert=CheXpert)

    extra_kwargs, extra_val_kwargs = {}, {}

    if args.db in ["cifar10", "cifar100", "cub200", "utzappos"]:
        extra_kwargs["download"] = True
        extra_val_kwargs["download"] = True
    if args.db in ["cifar10", "cifar100", "chexpert"]:
        extra_val_kwargs["train"] = False
    if args.db in ["cub200", "utzappos"]:
        extra_val_kwargs["split"] = "test"
    if args.db == "imagenet100":
        extra_val_kwargs["split"] = "val"
    if args.db == "chexpert":
        if args.chexpert_label is None:
            raise ValueError("CheXpert class unspecified. Please set --chexpert_label <label>.")
        extra_kwargs["labels"] = args.chexpert_label
        extra_val_kwargs["labels"] = args.chexpert_label
    dataset_cls = dataset2cls[args.db]

    train_dataset = dataset_cls(args.root, transform=transform["train"], **extra_kwargs)
    val_dataset = dataset_cls(args.root, transform=transform["val"], **extra_val_kwargs)

    return train_dataset, val_dataset