import collections
import os, tqdm
import os.path
import pprint
import numpy as np
import logging
import pandas as pd
from PIL import Image
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision import transforms
from datasets.prior import DatasetWithPrior
from pathlib import Path

class CheXpert(Dataset, DatasetWithPrior):
    """CheXpert Dataset (code adapted from https://github.com/mlmed/torchxrayvision)
    see "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison", Irvin et al., AAAI 2019

    Dataset must be downloaded here: https://stanfordmlgroup.github.io/competitions/chexpert/
    `root` == `/path/to/CheXpert-v1.0-small` unzip directory

    Warning: this dataset is multi-label (one sample can have several labels). It is one-hot encoded by a vector with
    potentially several 1's. Classical supervised approach: sum of 14 BCE (one for each label).
    """
    PATHOLOGIES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                   "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                   "Pleural Other", "Fracture", "Support Devices"]

    @property
    def prior_path(self):
        return os.path.join(Path(__file__).parent.parent.resolve(), "data", "chexpert", "chexpert_prior.npz")

    def __init__(self, root: str, train: bool=True,
                 views: [str, List[str]]="*", labels: str="all",
                 transform=None, target_transform=None,
                 weaklabels: bool=False):
        """
        :param root: str, path to images folder
        :param train: bool
        :param views: str or List[str], image views to load
        :param transform: callable, img transformation
        :param target_transform: callable
        """
        super(CheXpert, self).__init__()

        if weaklabels is True:
            self._build_prior()

        if isinstance(views, str): views = [views]
        assert set(views) <= {"PA", "AP", "*"}, "Unknown views: {}".format(views)
        if isinstance(labels, str): labels = [labels]

        if labels is not None:
            assert set(labels) <= set(self.PATHOLOGIES) | {"all"}, "Got {}".format(labels)

        self.pathologies = sorted(self.PATHOLOGIES)

        self.root = root
        self.return_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.views = views
        self.split = "train" if train else "valid"
        self._load_metadata()

    def _load_metadata(self):
        self.csvpath = os.path.join(self.root, self.split+".csv")
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)
        self.samples = self.csv.Path.map(lambda p: os.path.join(self.root, p.replace("CheXpert-v1.0-small/",""))).values
        # Assign view column
        self.csv["view"] = self.csv["Frontal/Lateral"]
        # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]
        # Rename Lateral with L
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"})

        self.limit_to_selected_views()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology != "Support Devices":
                self.csv.loc[healthy, pathology] = 0
            mask = self.csv[pathology]
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        if self.return_labels is not None and "all" not in self.return_labels:
            self.labels = self.labels[:, [(p in self.return_labels) for p in self.pathologies]]

        # patientid
        self.csv["patientid"] = self.csv.Path.str.extract(r"patient([0-9]*)").values

    def string(self):
        return self.__class__.__name__ + \
               " num_samples={} views={} tf={}".format(len(self), self.views, self.transform)

    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def check_paths_exist(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError("%s must be a directory"%self.root)
        if not os.path.isfile(self.csvpath):
            raise FileNotFoundError("%s must be a file"%self.csvpath)

    def build_prior(self, batch_size=128, num_workers=10):
        """
            Build CheXpert prior based on GLoRIA's representation.
        """
        if not os.path.isfile(self.prior_path) and self.split == "train":
            try:
                from gloria import load_img_classification_model
            except ModuleNotFoundError:
                raise ModuleNotFoundError("GloRIA is not installed. "
                                          "Check https://github.com/marshuang80/gloria.")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_img_classification_model("gloria_resnet18", device=device)

            current_tf = self.transform

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(0.5024, 0.2898)])
            
            train_loader = DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                                      pin_memory=True, sampler=SequentialSampler(self))

            priors, labels = [], []
            bar = tqdm.tqdm(desc="chexpert encoding", total=(len(train_loader) * batch_size))
            for samples, targets in train_loader:
                samples = samples.to(device)
                labels.extend(targets.cpu().detach().numpy())
                prior = model.img_encoder(samples)
                priors.extend(prior.cpu().detach().numpy())
                bar.update()
            priors = np.array(priors)
            labels = np.array(labels)
            np.savez(os.path.splitext(self.prior_path)[0], prior=priors, labels=labels)

            self.transform = current_tf

    def load_img(self, img):
        return Image.open(img).convert('L')

    def limit_to_selected_views(self):
        """This function is called to filter the
        images by view based on the values in .csv['view']
        """
        if type(self.views) is not list:
            self.views = [self.views]
        if '*' in self.views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        # missing data is unknown
        self.csv.view.fillna("UNKNOWN", inplace=True)

        if "*" not in self.views:
            self.csv = self.csv[self.csv["view"].isin(self.views)] # Select the view

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.load_img(self.samples[idx])
        target = self.labels[idx]
        if self.prior is not None:
            target = self.prior[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_labels:
            target = None
        sample = img
        if self.transform:
            sample = self.transform(img)
        return sample, target

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()