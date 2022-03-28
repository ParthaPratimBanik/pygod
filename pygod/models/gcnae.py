# -*- coding: utf-8 -*-
""" Graph Convolutional Network Autoencoder
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch
import torch.nn.functional as F
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import GCN
from ..utils.metric import eval_roc_auc


class GCNAE(BaseDetector):
    """
    Vanila Graph Convolutional Networks Autoencoder

    See :cite:`yuan2021higher` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``0``.
    num_layers : int, optional
        Total number of layers in autoencoders. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    contamination : float, optional
        Valid in (0., 0.5). The proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision
        function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``0``.
    verbose : bool
        Verbosity mode. Turn on to print out log information.
        Default: ``False``.

    Examples
    --------
    >>> from pygod.models import GCNAE
    >>> model = GCNAE()
    >>> model.fit(data)
    >>> prediction = model.predict(data)
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=100,
                 gpu=0,
                 verbose=False):
        super(GCNAE, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act

        # training param
        self.lr = lr
        self.epoch = epoch
        if gpu >= 0 and torch.cuda.is_available():
            self.device = 'cuda:{}'.format(gpu)
        else:
            self.device = 'cpu'

        # other param
        self.verbose = verbose
        self.model = None

    def fit(self, G):
        """
        Description
        -----------
        Fit detector with input data.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        x, edge_index, labels = self.process_graph(G)

        self.model = GCN(in_channels=x.shape[1],
                         hidden_channels=self.hid_dim,
                         num_layers=self.num_layers,
                         out_channels=x.shape[1],
                         dropout=self.dropout,
                         act=self.act).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        for epoch in range(self.epoch):
            self.model.train()
            x_ = self.model(x, edge_index)
            loss = F.mse_loss(x_, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            score = torch.mean(F.mse_loss(x_, x, reduction='none'), dim=1)

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, loss.item()), end='')
                if labels is not None:
                    auc = eval_roc_auc(labels, score.detach().cpu().numpy())
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = score.detach().cpu().numpy()
        self._process_decision_scores()
        return self

    def decision_function(self, G):
        """
        Description
        -----------
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        self.model.eval()

        x, edge_index, _ = self.process_graph(G)

        x_ = self.model(x, edge_index)
        outlier_scores = torch.mean(F.mse_loss(x_, x, reduction='none'),
                                    dim=1).detach().cpu().numpy()
        return outlier_scores

    def process_graph(self, G):
        """
        Description
        -----------
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        edge_index : torch.Tensor
            Edge list of the graph.
        y : torch.Tensor
            Labels of nodes.
        """
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        if hasattr(G, 'y'):
            y = G.y
        else:
            y = None

        return x, edge_index, y
