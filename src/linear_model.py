from abc import ABC
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np

def encode_dataset(loader: DataLoader, model: nn.Module, gpu: int=0) -> [np.ndarray, np.ndarray]:
    X, y = [], []
    for images, target in loader:
        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        # compute output
        output = model(images)
        X.extend(output.view(len(output), -1).detach().cpu().numpy())
        y.extend(target.detach().cpu().numpy())
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_linear(X_train, y_train, cv=3,
                 scoring="balanced_accuracy",
                 batch_size=512, num_workers=10):
    model = LinearClassifier(batch_size=batch_size, lr=0.1, epochs=300)
    param_grid = {"weight_decay": [0, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}
    model = GridSearchCV(model, cv=cv, param_grid=param_grid,
                         scoring=scoring, n_jobs=num_workers)
    model.fit(X_train, y_train)
    return model

class LinearEstimator(ABC, BaseEstimator):
    def __init__(self, lr=0.1, batch_size=128, epochs=300, momentum=0.9,
                 weight_decay=0.0, val_fraction=0.1, tol=1e-4):
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.val_fraction = val_fraction
        self.tol = tol
        self.display_freq = self.epochs // 10

    class ArrayDataset(Dataset):
        def __init__(self, X, y=None, transform=None):
            self.transform = transform
            if transform is None:
                self.transform = lambda x: x
            if y is not None:
                assert len(X) == len(y), "Wrong shape"
            self.X, self.y = X, y
        def __getitem__(self, i):
            if self.y is not None:
                return self.transform(self.X[i]), self.y[i]
            return self.transform(self.X[i])
        def __len__(self):
            return len(self.X)

class LinearClassifier(LinearEstimator, ClassifierMixin):
    """
        Implements linear classifier in a scikit-learn fashion trained with SGD on CUDA (with PyTorch).
        It is scalable and faster than sklearn.SGDClassifier runt on CPU.
        It implements a .fit(), .predict() and .predict_proba() method
    """
    def __init__(self, lr=0.1, batch_size=128, epochs=300, momentum=0.9,
                 weight_decay=0.0, val_fraction=0.1, tol=1e-4, transform=None):
        super().__init__(lr, batch_size, epochs, momentum, weight_decay, val_fraction, tol)
        self.classifier = None
        self.transform = transform

    class LinearModel(nn.Module):
        """Linear classifier"""
        def __init__(self, feat_dim, num_classes=10):
            super().__init__()
            self.fc = nn.Linear(feat_dim, num_classes)

        def forward(self, features):
            return self.fc(features)

    def predict(self, X):
        check_is_fitted(self)

        self.classifier.eval()
        loader = DataLoader(self.ArrayDataset(X, transform=self.transform), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
                out = self.classifier(x).detach().cpu().numpy()
                outputs.extend(out.argmax(axis=1))
        return np.array(outputs)

    def predict_proba(self, X):
        check_is_fitted(self)

        self.classifier.eval()
        loader = DataLoader(self.ArrayDataset(X, transform=self.transform), batch_size=self.batch_size)
        outputs = []
        for x in loader:
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.classifier(x)
            outputs.extend(torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy())
        return np.array(outputs)

    def fit(self, X, y):
        X, y = check_X_y(X, y, multi_output=True)

        self.classes_ = unique_labels(y)
        # build data loaders
        train_loader = DataLoader(self.ArrayDataset(X, y, transform=self.transform),
                                  batch_size=self.batch_size, shuffle=True)
        self.num_features = X.shape[1]
        if self.transform is not None and len(X) > 0:
            self.num_features = len(self.transform(X[0]))

        # build model and criterion
        self.classifier = self.LinearModel(self.num_features, num_classes=len(self.classes_))
        if torch.cuda.is_available():
            self.classifier = self.classifier.to('cuda')
        criterion = nn.CrossEntropyLoss()

        # build optimizer
        optimizer = torch.optim.SGD(self.classifier.parameters(),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

        # training routine
        losses = []
        stopping_criterion = None
        patience = 10
        acc = 0.0
        for epoch in range(1, self.epochs + 1):
            # train for one epoch
            loss, acc = self.train(train_loader, self.classifier, criterion, optimizer)
            scheduler.step(loss)
            losses.append(loss)
            if len(losses) > 2 * patience:
                stopping_criterion = np.max(np.abs(np.mean(losses[-patience:]) - losses[-patience:]))
                if stopping_criterion < self.tol: # early-stopping
                    break
        losses = np.array(losses)
        if (np.max(np.abs(np.mean(losses[-patience:]) - losses[-patience:])) > self.tol):
            print("Warning: max iter reached before clear convergence", flush=True)
        return self

    def train(self, train_loader, classifier, criterion, optimizer):
        """one epoch training"""
        classifier.train()
        losses = []
        top1 = []
        for idx, (features, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            # compute loss
            output = classifier(features)
            loss = criterion(output, labels.long())
            # update metric
            losses.append(loss.detach().cpu().numpy())
            acc1 = self.accuracy(output, labels)
            top1.append(acc1)
            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return np.mean(losses), np.mean(top1)

    @staticmethod
    def accuracy(y_pred, y):
        if len(y_pred.shape) == 1:
            y_pred = (y_pred > 0)
        y_pred = y_pred.data.max(dim=1)[1]
        return accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())