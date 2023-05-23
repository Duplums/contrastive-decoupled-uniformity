import os
from itertools import compress
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import pandas as pd
import numpy as np
from pathlib import Path
from datasets.prior import DatasetWithPrior

class CUB(ImageFolder, DatasetWithPrior):
    """
    cf. Catherine Wah et al, The caltech-ucsd birds-200-2011 dataset, 2011
    200 bird categories with 11788 images.
    This dataset contains 312 additional binary attributes considered as meta-data.
    It can be used as "labels" in a self-supervised setting (only for training).
    The training/test split follows the official one.
    """
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    @property
    def prior_path(self):
        return os.path.join(Path(__file__).parent.parent.resolve(), "data", "cub", "cub_prior.npz")

    def __init__(self, root, transform=None, target_transform=None,
                 split="train", download=True, **kwargs):
        """
        :param root: str, path to images folder
        :param transform: callable, img transformation
        :param target_transform: callable
        :param split: str, either "train" or "test"
        :param download, bool
        :param kwargs: given to super()
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        """
            Sets `attrs`, `imgs` and `samples` attributes
        """
        weaklabels = self.kwargs.pop("weaklabels", None)
        # Checks images repo and find all img paths
        super().__init__(os.path.join(self.root, self.base_folder),
                         self.transform, self.target_transform, **self.kwargs)

        # Defines training/test split from the official one
        train_test_split_pth = os.path.join(self.root, "CUB_200_2011", "train_test_split.txt")
        img_pth = os.path.join(self.root, "CUB_200_2011", "images.txt")
        if not os.path.exists(train_test_split_pth) or not os.path.exists(img_pth):
            raise FileNotFoundError("Missing %s or %s in CUB dataset"%(train_test_split_pth, img_pth))
        # "0" == test, "1" == train
        train_test_split = pd.read_csv(train_test_split_pth, sep=" ", names=["id", "split"])
        img_pth = pd.read_csv(img_pth, sep=" ", names=["id", "path"])
        pth_split = pd.merge(train_test_split, img_pth, on="id", how="inner")
        this_split = list(pth_split[pth_split.split.eq(self.split == "train")].path)

        filter = np.array(["/".join(pth.split("/")[-2:]) in this_split for (pth, _) in self.samples], dtype=np.bool)
        assert filter.sum() == len(this_split), "Corrupted CUB data-set: " \
                                                "images missing or corrupted train_test_split.txt"
        self.samples = list(compress(self.samples, filter))
        self.imgs = self.samples

        if self.split == "train":
            assert len(self) == 5994
        else:
            assert len(self) == 5794

        if weaklabels is True:
            self._build_prior()

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def _build_prior(self):
        if not os.path.isfile(self.prior_path) and self.split == "train":
            attr_path = os.path.join(os.path.dirname(self.prior_path), "meta_data_bin_train.csv")
            if not os.path.exists(attr_path):
                raise FileNotFoundError("Inconsistent github repo.")
            attr = pd.read_csv(attr_path, sep=",")
            # generate N X M matrix where N == len(train) and M == # attributes
            imgs_df = pd.DataFrame(self.imgs, columns=["path", "class"])
            imgs_df.loc[:, "path"] = imgs_df.path.apply(lambda p: "/".join(p.split("/")[-2:]))
            attr = pd.merge(imgs_df, attr, on=["path"], how="left", sort=False)
            attr_cols = [i for i in attr.columns if 'attr_val' in i]
            attr = attr[attr_cols].to_numpy(dtype=np.float32)
            assert len(attr) == len(self)
            np.savez(os.path.splitext(self.prior_path)[0], prior=attr, labels=self.targets)
        super()._build_prior()

    def __getitem__(self, idx):
        sample, label = super().__getitem__(idx)
        if self.prior is not None:
            label = self.prior[idx]
        return sample, label
