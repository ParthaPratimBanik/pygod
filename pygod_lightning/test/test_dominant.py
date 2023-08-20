import tqdm
import torch
import argparse
from random import choice
from pygod.detector import *
from pygod.utils import load_data

from torch_geometric.seed import seed_everything
from pygod_lightning.dataset import DataSet
from pygod_lightning.nn import DOMINANTBase

def main(args):
    ## Checking Training Result on pygodv100 implementation
    model = DOMINANT(hid_dim=choice(hid_dim),
                    weight_decay=weight_decay,
                    dropout=choice(dropout),
                    lr=choice(lr),
                    epoch=epoch,
                    gpu=gpu,
                    weight=choice(alpha),
                    batch_size=batch_size,
                    num_neigh=num_neigh)
    data = load_data(args.dataset)
    model.fit(data)
    # score = model.decision_score_

    ## Checking Training Result on pytorch.lightning implementation
    modelPL = DOMINANTBase(hid_dim=choice(hid_dim),
                            weight_decay=weight_decay,
                            dropout=choice(dropout),
                            lr=choice(lr),
                            epoch=epoch,
                            gpu=gpu,
                            weight=choice(alpha),
                            batch_size=batch_size,
                            num_neigh=num_neigh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon, "
                             "inj_flickr, weibo, reddit, disney, books, "
                             "enron]. Default: inj_cora")
    args = parser.parse_args()

    ## Declare the parameters
    dropout = [0, 0.1, 0.3]
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01
    batch_size = 0
    num_neigh = -1
    epoch = 300
    gpu = args.gpu
    hid_dim = [32, 64, 128, 256]
    alpha = [0.8, 0.5, 0.2]

    main(args)