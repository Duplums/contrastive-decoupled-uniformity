import torch
import torch.nn as nn
import numpy as np
from src.util import RBFKernel, CosineKernel
import torch.functional as func

class DecoupledUniformity(nn.Module):
    """
    Build a DNN model with a 2-layers MLP as projector (similar to SimCLR).
    """
    def __init__(self, base_encoder, dim=2048, proj_dim=256, first_conv=True):
        """
        dim: int, default 2048
            Projection head hidden dimension
        proj_dim: int, default 256
            Projection dimension
        first_conv: bool, default True
            If False, set first conv kernel to 3x3 and remove first max pooling.

        """
        super(DecoupledUniformity, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        if not first_conv: # for small-scale images
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.encoder.maxpool = nn.Identity()

        # build a 2-layers projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
                                    nn.ReLU(True),
                                    nn.Linear(prev_dim, dim),
                                    nn.ReLU(True),
                                    nn.Linear(dim, proj_dim))

    def forward(self, x):
        """
        Input:
            x: batch of images, shape bsize*nviews x channels x spatial dimensions
        Output:
            z: latent representation
        """

        # compute features for all images
        z = self.encoder(x)
        return z


class DecoupledUniformityLoss(nn.Module):
    """
        Computes uniformity loss between centroids.
    """
    def __init__(self, kernel: str=None, lambda_: float=0.01, t: float=2., **kernel_kwargs):
        """
        :param kernel: {'rbf', 'cosine'}, default None
            If weak labels are given, the kernel is used to compute centroid estimator mu_X = E f(X)
        :param lambda: float, default None
            Reg. coefficient for centroid estimation mu_X = (K + lambda * I)**-1 * K.
        :param t: float, default 2
            Uniformity scaling corresponding to hyper-sphere radius
        """
        super().__init__()
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.lambda_ = lambda_
        self.t = t
        self.metrics = dict()
        if self.kernel is not None:
            if self.kernel == "rbf":
                sigma = self.kernel_kwargs.get("sigma")
                self.sigma = sigma
                assert sigma is not None, "\sigma must be provided with RBF kernel."
                self.kernel = RBFKernel(gamma=1/(2*sigma**2))
            elif self.kernel == "cosine":
                self.kernel = CosineKernel()
            else:
                raise ValueError("Unknown kernel: %s"%self.kernel)

    def forward(self, z, prior=None):
        """
        :param z: torch.Tensor, shape (bsize, nviews, feature_dim)
            Input feature vectors
        :param prior: torch.Tensor, shape (bsize, *)
            Auxiliary attributes
        :return: torch.Tensor(float)
            Decoupled Uniformity loss
        """
        # Map feature vectors to hyper-sphere using l2-normalization
        z = func.normalize(z, p=2, dim=-1) # dim [bsize, nviews, *]
        centroids = self._get_centroids(z, prior)
        align = self.align_loss(centroids)
        unif = self.uniform_loss(centroids)
        self.metrics['unif'] = unif.detach().cpu().numpy()
        self.metrics['align'] = align.detach().cpu().numpy()
        return unif

    def _get_centroids(self, z: torch.Tensor, prior: torch.Tensor=None):
        assert len(z.shape) == 3, "Incorrect tensor shape, got {}".format(z.shape)
        bs, nviews, dim = z.shape
        if self.kernel is not None and prior is not None:
            z = torch.mean(z, dim=1)
            if len(prior.shape) == 1:
                prior = prior.unsqueeze(1).float()
            lambda_ = self.lambda_ or 1./np.sqrt(bs)
            sim = self.kernel(prior.detach(), prior.detach())
            alpha = (torch.inverse(sim + lambda_ * torch.eye(bs, device=z.device)) @ sim)
            z = alpha.detach() @ z
        else:
            z = torch.mean(z, dim=1)
        return z

    def uniform_loss(self, z: torch.Tensor):
        """
        :param z: shape bsize x feature_dim
        :return: Uniformity between centroids
        """
        return torch.pdist(z, p=2).pow(2).mul(-self.t).exp().mean().log()

    def align_loss(self, z: torch.Tensor, alpha=2):
        """
        :param z: shape bsize x feature_dim
        :return: Alignment of centroids (the lower the better)
       """
        return -torch.norm(z, p=2, dim=1).pow(alpha).mean()