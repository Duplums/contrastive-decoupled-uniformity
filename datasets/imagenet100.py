from torchvision.datasets import ImageNet
from datasets.prior import DatasetWithPrior
import os
from pathlib import Path

class ImageNet100(ImageNet, DatasetWithPrior):
    """
        100-classes subset of ImageNet introduced in [1]
        [1] Contrastive Multiview Coding, Tian, Krishnan, Isola, ECCV 2020
    """

    def __init__(self, *args, **kwargs):
        weaklabels = kwargs.pop("weaklabels", None)
        super().__init__(*args, **kwargs)
        if weaklabels is True:
            self._build_prior()

    @property
    def prior_path(self):
        return os.path.join(Path(__file__).parent.parent.resolve(), "data", "imagenet100", "imagenet100_prior.npz")

    def _extract_classes(self, all_classes=None):
        """
        :param all_classes: list of all class names used to check if filtered class are inside them
        :return: a list of 100 class names extracted from 'imagenet100.txt' sorted alphabetically
        """
        pth = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "imagenet100", "imagenet100.txt")
        if not os.path.isfile(pth):
            url = "https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt"
            try:
                from urllib.request import urlretrieve
                urlretrieve(url, pth)
            except Exception:
                raise FileNotFoundError("File %s not found. Please download it from %s"%(pth, url))
        with open(pth, 'r') as f:
            classes = f.read().split("\n")
            if classes[-1] == '':
                classes = classes[:-1]
            classes.sort()
            if all_classes is not None:
                assert set(classes) <= set(all_classes), \
                    "Some classes are unknown: {}".format(set(classes) - set(all_classes))
        return classes

    def _find_classes(self, dir):
        """
            Finds the class folders in a dataset.

            Args:
                dir (string): Root directory path.

            Returns:
                tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

            Ensures:
                No class is a subdirectory of another.
        """
        # Filter only the selected classes from "Contrastive Multiview Coding", Y. Tian, ECCV 2020
        all_classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes = self._extract_classes(all_classes)
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, idx):
        sample, label = super().__getitem__(idx)
        if hasattr(self, "prior") and self.prior is not None:
            label = self.prior[idx]
        return sample, label
