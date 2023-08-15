from datetime import datetime
import os
import tqdm
import torch
import argparse
import warnings
from pygod.metric import *
from pygod.utils import load_data
from pygod.detector import *

def run_model(hid_dim, dropout, weight, weight_decay, lr, args):

    model = DOMINANT(
                hid_dim=hid_dim,
                weight_decay=weight_decay,
                dropout=dropout,
                lr=lr, # it varies result a lot
                gpu=args.gpu,
                wight=weight)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    # print("date and time =", dt_string)
    fileName = dir_name + "/" + dt_str + ".txt"

    results = "*"*50
    results += "\nStart Date-Time: " + now.strftime("%d/%m/%Y, %H:%M:%S") + "\n"
    results += "\nDataset: " + args.dataset + ", Model: " + model.__class__.__name__ + "\n"
    results += "\nHyper-Parameter Setting:"
    results += "\nhid_dim=%d\ndropout=%f" % (hid_dim, dropout)
    results += "\nweight=" + str(weight)
    results += "\nweight_decay=%f\nlr=%f\ngpu=%d\n" % (weight_decay, lr, args.gpu)

    AUC_TH = 0.80
    count_auc_gt80 = 0
    
    auc, ap, rec = [], [], []

    count_inter=0
    for _ in tqdm.tqdm(range(num_trial)):
        data = load_data(args.dataset)

        model.fit(data)
        score = model.decision_score_

        y = data.y.bool()
        k = sum(y)

        if torch.isnan(score).any():
            warnings.warn('contains NaN, skip one trial.')
            continue

        auc_val = eval_roc_auc(y, score)
        # print("auc: ", auc_val)

        if auc_val > AUC_TH:
            count_auc_gt80 += 1
        
        auc.append(auc_val)
        ap.append(eval_average_precision(y, score))
        rec_val = eval_recall_at_k(y, score, k)
        if isinstance(rec_val, torch.Tensor):
            rec_val = rec_val.tolist()
        rec.append(rec_val)

        count_inter += 1

        if count_inter > 5:
            if count_auc_gt80 == 0:
                break

    auc = torch.tensor(auc)
    ap = torch.tensor(ap)
    rec = torch.tensor(rec)

    print(args.dataset + " " + model.__class__.__name__ + " " +
          "AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\t"
          "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc),
                                                  torch.std(auc),
                                                  torch.max(auc),
                                                  torch.mean(ap),
                                                  torch.std(ap),
                                                  torch.max(ap),
                                                  torch.mean(rec),
                                                  torch.std(rec),
                                                  torch.max(rec)))
    
    results += "\nROC-AUC:"
    results += "\nAUC:    %.4f±%.4f (%.4f)" % (torch.mean(auc), torch.std(auc), torch.max(auc))
    results += "\nAP:     %.4f±%.4f (%.4f)" % (torch.mean(ap), torch.std(ap), torch.max(ap))
    results += "\nRecall: %.4f±%.4f (%.4f)" % (torch.mean(rec), torch.std(rec), torch.max(rec))
    results += "\n"
    results += "*"*50

    with open(fileName, 'w') as f:
        f.write(results)


def main(args):
    print("\ndataset: ", args.dataset)
    print("gpu: ", args.gpu)
    print("model_name: ", args.model)

    dropout = [0.0, 0.1, 0.3, 0.5, 0.7]
    lr = [0.004, 0.01, 0.05, 0.10]      # lr brings change on AUC
    weight_decay = [0.0, 0.01]
    hid_dim = [8, 12, 16, 32, 48, 64, 128, 256, 512, 1024]
    weight = [None, 0.5]

    for hd in hid_dim:
        for do in dropout:
            for wgt in weight:
                for wd in weight_decay:
                    for lr_val in lr:
                        run_model(hid_dim=hd,
                                  dropout=do,
                                  weight=wgt,
                                  weight_decay=wd,
                                  lr=lr_val,
                                  args=args)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dominant",
                        help="supported model: [lof, if, mlpae, scan, radar, "
                             "anomalous, gcnae, dominant, done, adone, "
                             "anomalydae, gaan, guide, conad]. "
                             "Default: dominant")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon, "
                             "inj_flickr, weibo, reddit, disney, books, "
                             "enron]. Default: inj_cora")
    args = parser.parse_args()

    # global setting
    num_trial = 20

    # make result_directory
    now = datetime.now()
    # dd/mm/YY H:M:S
    dir_name = "results\\" + now.strftime("%d_%m_%Y_%H_%M")
    # print("date and time =", dt_string)
    os.makedirs(dir_name)

    main(args)