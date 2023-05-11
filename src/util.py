import torch
import torch.nn as nn
import numpy as np
import torch.functional as func

class RBFKernel(nn.Module):
    """
        Compute the rbf (gaussian) kernel between X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.
        Parameters
        ----------
        X : ndarray array of shape (n_samples_X, n_features)
        Y : ndarray array of shape (n_samples_Y, n_features)
        gamma : float
        Returns
        -------
        kernel_matrix : array of shape (n_samples_X, n_samples_Y)
        """
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, X, Y):
        XX = (X * X).sum(axis=1)[:, np.newaxis]
        if Y is X:
            YY = XX.T
        else:
            YY = (Y * Y).sum(axis=1)[np.newaxis, :]
        eucl_dist2 = XX + YY - 2 * (X @ Y.T)
        K = torch.exp(-self.gamma * eucl_dist2)
        return K

class CosineKernel(nn.Module):
    """Compute cosine similarity between samples in X and Y.
       Cosine similarity, or the cosine kernel, computes similarity as the
       normalized dot product of X and Y:
           K(X, Y) = <X, Y> / (||X||*||Y||)
       On L2-normalized data, this function is equivalent to linear_kernel.
       Parameters
       ----------
       X : ndarray or sparse array, shape: (n_samples_X, n_features)
           Input data.

       Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
           Input data. If ``None``, the output will be the pairwise
           similarities between all samples in ``X``.
        Returns
        -------
        kernel matrix : array
            An array with shape (n_samples_X, n_samples_Y).
    """

    def forward(self, X, Y):
        X_norm = func.normalize(X, p=2, dim=-1)
        if Y is X:
            Y_norm = X_norm
        else:
            Y_norm = func.normalize(Y, p=2, dim=-1)
        K = X_norm @ Y_norm.T
        return K