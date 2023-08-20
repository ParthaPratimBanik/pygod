import argparse
from pygod.utils import load_data
from dataset import DataSet

def main(args):
    pgv100_data = load_data(args.dataset)
    print("\npgv100_data: ", pgv100_data)
    print("pgv100_data type: ", type(pgv100_data))
    # print("x= ", pgv100_data.x)
    # print("y= ", pgv100_data.y)
    # for i in pgv100_data:
    #     print(i)
    
    # test on Dataset class
    # dsc = DatasetBase(args.dataset)
    # print("length: ", len(dsc))
    # for sample in dsc:
    #     print(sample)

    dl_obj = DataSet(name=args.dataset, batch_size=0)
    dl_obj.prepare_data()
    dl_obj.setup()
    # for idx in range(len(dl_obj.db)):
    #     print(dl_obj.db[idx])

    train_loader = dl_obj.train_dataloader()
    print("train_loader: ", train_loader)
    print("train_loader type: ", type(train_loader))
    for bidx, batch in enumerate(train_loader):
        print("\n.%d"%(bidx))
        print("batch: ",batch)

    # for bidx, batch in enumerate(train_loader):
    #     print("\n%d."%(bidx))
    #     print("batch.x: ", batch["x"])
    #     print("batch.y: ", batch["y"])

    # tonl = TestOnNL(name=args.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon, "
                             "inj_flickr, weibo, reddit, disney, books, "
                             "enron]. Default: inj_cora")
    args = parser.parse_args()

    main(args)