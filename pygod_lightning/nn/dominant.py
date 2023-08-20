import math
import torch
# import torch.nn as nn
from torch_geometric.nn import GCN
# from torch_geometric.utils import to_dense_adj

from .decoder import DotProductDecoder
from .functional import double_recon_loss

import lightning.pytorch as pl


class DOMINANTBase(pl.LightningModule):
    """
    Deep Anomaly Detection on Attributed Networks

    DOMINANT is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The reconstruction mean squared
    error of the decoders are defined as structure anomaly score and
    attribute anomaly score, respectively.

    See :cite:`ding2019deep` for details.

    Parameters
    ----------
    in_dim : int
        Input dimension of model.
    hid_dim :  int
       Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
       Total number of layers in model. A half (floor) of the layers
       are for the encoder, the other half (ceil) of the layers are
       for decoders. Default: ``4``.
    dropout : float, optional
       Dropout rate. Default: ``0.``.
    act : callable activation function or None, optional
       Activation function if not None.
       Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to apply sigmoid to the structure reconstruction.
        Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    **kwargs : optional
        Additional arguments for the backbone.
    """

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 weight_decay=0.,
                 **kwargs):
        super(DOMINANTBase, self).__init__()

        # split the number of layers for the encoder and decoders
        assert num_layers >= 2, \
            "Number of layers must be greater than or equal to 2."
        encoder_layers = math.floor(num_layers / 2)
        decoder_layers = math.ceil(num_layers / 2)

        self.shared_encoder = backbone(in_channels=in_dim,
                                       hidden_channels=hid_dim,
                                       num_layers=encoder_layers,
                                       out_channels=hid_dim,
                                       dropout=dropout,
                                       act=act,
                                       **kwargs)

        self.attr_decoder = backbone(in_channels=hid_dim,
                                     hidden_channels=hid_dim,
                                     num_layers=decoder_layers,
                                     out_channels=in_dim,
                                     dropout=dropout,
                                     act=act,
                                     **kwargs)

        self.struct_decoder = DotProductDecoder(in_dim=hid_dim,
                                                hid_dim=hid_dim,
                                                num_layers=decoder_layers - 1,
                                                dropout=dropout,
                                                act=act,
                                                sigmoid_s=sigmoid_s,
                                                backbone=backbone,
                                                **kwargs)

        self.weight_decay = weight_decay
        self.emb = None

    def forward(self, x, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input attribute embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        x_ : torch.Tensor
            Reconstructed attribute embeddings.
        s_ : torch.Tensor
            Reconstructed adjacency matrix.
        """
        # encode feature matrix
        self.emb = self.shared_encoder(x, edge_index)

        # reconstruct feature matrix
        x_ = self.attr_decoder(self.emb, edge_index)

        # decode adjacency matrix
        s_ = self.struct_decoder(self.emb, edge_index)

        return x_, s_
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        batch_size = batch.batch_size
        node_idx = batch.n_id

        x = batch.x.to(self.device)
        s = batch.s.to(self.device)
        edge_index = batch.edge_index.to(self.device)

        x_, s_ = self(x, edge_index)

        score = double_recon_loss(x[:batch_size],
                                     x_[:batch_size],
                                     s[:batch_size, node_idx],
                                     s_[:batch_size],
                                     self.weight)

        loss = torch.mean(score)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        return optimizer
