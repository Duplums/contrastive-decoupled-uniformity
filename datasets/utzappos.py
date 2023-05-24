import os
from itertools import compress
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
import pandas as pd
import numpy as np
from pathlib import Path
from datasets.prior import DatasetWithPrior

class UTZappos(ImageFolder, DatasetWithPrior):
    """
    cf. Fine-Grained Visual Comparisons with Local Learning, A. Yu & K. Grauman, CVPR 2014
    7 auxiliary attributes with 50025 shoes images.
    These attributes can be binarized to obtain 126 binary attributes.
    Following [1], 21 shoe categories are used as labels and 126 binary attributes as prior.
    The training/test split follows the one defines in [1].
    [1] Tsai et al., Conditional Contrastive Learning with Kernel, ICLR 2022
    """
    base_folder = 'ut-zap50k-images/images'
    url_data = 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip'
    url_images = 'https://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip'
    filenames = dict(data='ut-zap50k-data.zip', images='ut-zap50k-images.zip')

    @property
    def prior_path(self):
        return os.path.join(Path(__file__).parent.parent.resolve(), "data", "utzappos", "utzappos_prior.npz")

    _classes = ['Ankle', 'Athletic', 'Boat Shoes', 'Boot', 'Clogs and Mules',
                'Crib Shoes', 'Firstwalker', 'Flat', 'Flats', 'Heel', 'Heels',
                'Knee High', 'Loafers', 'Mid-Calf', 'Over the Knee', 'Oxfords',
                'Prewalker', 'Prewalker Boots', 'Slipper Flats', 'Slipper Heels',
                'Sneakers and Athletic Shoes']

    def __init__(self, root, transform=None, target_transform=None,
                 split="train", download=True, **kwargs):
        """
        :param root: str, path to images folder
        :param transform: callable, img transformation
        :param target_transform: callable
        :param split: str, either "train" or "test"
        :param download: bool
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
        kwargs = self.kwargs.copy()
        weaklabels = kwargs.pop("weaklabels", None)
        root = self.root
        # Checks images repo and find all img paths (modifies self.root internally)
        super().__init__(os.path.join(self.root, "ut-zap50k-images"),
                         self.transform, self.target_transform, **kwargs)
        self.root = root
        # Get training split defined in [1]
        attr_path = os.path.join(os.path.dirname(self.prior_path), "meta_data_bin_train.csv")
        if not os.path.exists(attr_path):
            raise FileNotFoundError("Corrupted repo, missing file: %s"%attr_path)
        train_attributes = pd.read_csv(attr_path, sep=",")
        extract_cid = lambda p: os.path.basename(p).replace(".jpg", "").replace(".", "-")
        train_cid= train_attributes["path"].apply(extract_cid)

        # Extract label data
        label_path = os.path.join(self.root, "ut-zap50k-data", "meta-data.csv")
        if not os.path.exists(label_path):
            raise FileNotFoundError("Labels not found in %s"%label_path)
        all_ids = pd.read_csv(label_path, sep=",")
        all_ids["train"] = all_ids["CID"].isin(train_cid)
        all_ids["test"] = ~all_ids["train"]
        imgs_df = pd.DataFrame(self.samples, columns=["path", "class"])
        imgs_df["CID"] = imgs_df["path"].apply(extract_cid)
        filter = imgs_df["CID"].isin(all_ids[all_ids[self.split]]["CID"]).to_numpy()
        assert filter.sum() == all_ids[self.split].sum(), "Corrupted UTZappos dataset"
        self.samples = list(compress(self.samples, filter))
        labels = pd.merge(imgs_df, all_ids, on=["CID"], how="left", validate="1:1")
        labels = labels[filter]
        self.samples = [(pth, labels["SubCategory"].values[i])
                        for i, (pth, _) in enumerate(self.samples)]

        classes = np.unique([c for (_, c) in self.samples])
        assert set(classes) <= set(UTZappos._classes)
        self.class_to_idx = {c: i for i, c in enumerate(self._classes)}
        self.classes = classes
        self.samples = [(pth, self.class_to_idx[c]) for (pth, c) in self.samples]
        self.targets = [s[1] for s in self.samples]

        if self.split == "train":
            assert len(self) == 35017
        else:
            assert len(self) == 15008

        if weaklabels is True:
            self._build_prior()

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            print(e)
            return False
        return True

    def _download(self):
        import zipfile
        root = self.root
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        self.root = root
        download_url(self.url_data, self.root, self.filenames['data'])
        download_url(self.url_images, self.root, self.filenames['images'])

        for filename in self.filenames.values():
            with zipfile.ZipFile(os.path.join(self.root, filename), 'r') as zip:
                zip.extractall(path=self.root)

    def _build_prior(self):
        if not os.path.isfile(self.prior_path) and self.split == "train":
            attr_path = os.path.join(os.path.dirname(self.prior_path), "meta_data_bin_train.csv")
            if not os.path.exists(attr_path):
                raise FileNotFoundError("Corrupted repo, missing file: %s"%attr_path)
            train_attributes = pd.read_csv(attr_path, sep=",")
            extract_cid = lambda p: os.path.basename(p).replace(".jpg", "").replace(".", "-")
            train_attributes["CID"] = train_attributes["path"].apply(extract_cid)
            imgs_df = pd.DataFrame(self.samples, columns=["path", "class"])
            imgs_df["CID"] = imgs_df.path.apply(extract_cid)
            train_attributes = pd.merge(imgs_df, train_attributes, on=["CID"], how="left", sort=False,
                                        validate="1:1")
            attr_cols = [i for i in train_attributes.columns if 'attr_val' in i]
            train_attributes = train_attributes[attr_cols].to_numpy(dtype=np.float32)
            np.savez(os.path.splitext(self.prior_path)[0], prior=train_attributes, labels=self.targets)
        super()._build_prior()

    def __getitem__(self, idx):
        sample, label = super().__getitem__(idx)
        if hasattr(self, "prior") and self.prior is not None:
            label = self.prior[idx]
        return sample, label
