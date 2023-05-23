import numpy as np
from abc import ABC, abstractmethod
import os


class DatasetWithPrior(ABC):
    """ Abstract class for any PyTorch dataset that uses a
        prior representation instead of target labels.
    """
    @property
    @abstractmethod
    def prior_path(self):
        pass

    def _build_prior(self):
        prior_path = self.prior_path
        if not os.path.isfile(prior_path):
            raise FileNotFoundError("Check %s" % prior_path)
        try:
            pth_loaded = np.load(prior_path)
            self.prior = pth_loaded["prior"]
            true_labels = pth_loaded["labels"]
        except Exception:
            raise ValueError("Check numpy array: %s" % prior_path)
        if len(self.prior.shape) == 3:
            self.prior = self.prior[:, 0]
        if len(true_labels.shape) == 3:
            true_labels = true_labels[:, 0]
        assert len(self.prior) == len(self), "Inconsistent # of samples"
        if hasattr(self, "targets"):
            assert np.all(true_labels.squeeze() == self.targets), "Inconsistent labels"
        print("Weak labels loaded.", flush=True)