from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from models.classifiers.nn_classifiers.nn_classifier import NNClassifier
from models.loss_functions import GeneralCrossEntropy
from utils.data_utils.tsptw_dataset import TSPTWDataloader
from utils.data_utils.pctsp_dataset import PCTSPDataloader
from utils.data_utils.pctsptw_dataset import PCTSPTWDataloader
from utils.data_utils.cvrp_dataset import CVRPDataloader
from utils.utils import set_device, count_trainable_params, batched_bincount, fix_seed

def main(args):
    #---------------
    # seed settings
    #---------------
    fix_seed(args.seed)

    #--------------
    # gpu settings
    #--------------
    use_cuda, device = set_device(args.gpu)

    #-------------------
    # model & optimizer
    #-------------------
    num_classes = 3 if args.problem == "pctsptw" else 2
    model = NNClassifier(problem=args.problem,
                         node_enc_type=args.node_enc_type,
                         edge_enc_type=args.edge_enc_type,
                         dec_type=args.dec_type,
                         emb_dim=args.emb_dim,
                         num_enc_mlp_layers=args.num_enc_mlp_layers,
                         num_dec_mlp_layers=args.num_dec_mlp_layers,
                         num_classes=num_classes,
                         dropout=args.dropout,
                         pos_encoder=args.pos_encoder)
    is_sequential = model.is_sequential
    if use_cuda:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # count number of trainable parameters
    num_trainable_params = count_trainable_params(model)
    print(f"num_trainable_params: {num_trainable_params}")
    with open(f"{args.model_checkpoint_path}/num_trainable_params.dat", "w") as f:
        f.write(str(num_trainable_params))

    # loss function
    if not is_sequential:
        assert args.loss_function != "seq_cbce", "Non-sequential model does not support the loss funtion: seq_cbce"
    loss_func = GeneralCrossEntropy(weight_type=args.loss_function, beta=args.cb_beta, is_sequential=is_sequential)

    #---------
    # dataset
    #---------
    if args.problem == "tsptw":
        train_dataset = TSPTWDataloader(args.train_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
        if args.valid_dataset_path is not None:
            valid_dataset = TSPTWDataloader(args.valid_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
    elif args.problem == "pctsp":
        train_dataset = PCTSPDataloader(args.train_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
        if args.valid_dataset_path is not None:
            valid_dataset = PCTSPDataloader(args.valid_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
    elif args.problem == "pctsptw":
        train_dataset = PCTSPTWDataloader(args.train_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
        if args.valid_dataset_path is not None:
            valid_dataset = PCTSPTWDataloader(args.valid_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
    elif args.problem == "cvrp":
        train_dataset = CVRPDataloader(args.train_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
        if args.valid_dataset_path is not None:
            valid_dataset = CVRPDataloader(args.valid_dataset_path, sequential=is_sequential, parallel=args.parallel, num_cpus=args.num_cpus)
    else:
        raise NotImplementedError

    #------------
    # dataloader
    #------------
    if is_sequential:
        def pad_seq_length(batch):
            data = {}
            for key in batch[0].keys():
                padding_value = True if key == "mask" else 0.0
                # post-padding
                data[key] = torch.nn.utils.rnn.pad_sequence([d[key] for d in batch], batch_first=True, padding_value=padding_value)
            pad_mask = torch.nn.utils.rnn.pad_sequence([torch.full((d["mask"].size(0), ), True) for d in batch], batch_first=True, padding_value=False)
            data.update({"pad_mask": pad_mask})
            return data
        collate_fn = pad_seq_length
    else:
        collate_fn = None
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=args.num_workers)
    if args.valid_dataset_path is not None:
        valid_dataloader = DataLoader(valid_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      collate_fn=collate_fn,
                                      num_workers=args.num_workers)

    #---------
    # metrics
    #---------
    macro_accuracy = MulticlassF1Score(num_classes=num_classes, average="macro")
    if use_cuda:
        macro_accuracy.to(device)

    #---------------
    # training loop
    #---------------
    best_valid_accuracy = 0.0
    model.train()
    with tqdm(range(args.epochs + 1)) as tq1:
        for epoch in tq1:
            #--------------------------
            # save the current weights
            #--------------------------
            # print(f"Epoch {epoch}: saving a model to {args.model_checkpoint_path}/model_epoch{epoch}.pth...", end="", flush=True)
            torch.save(model.cpu().state_dict(), f"{args.model_checkpoint_path}/model_epoch{epoch}.pth")
            model.to(device)
            # print("done.")

            #------------
            # validation
            #------------
            model.eval()
            with torch.no_grad():
                tq1.set_description(f"Epoch {epoch}")
                # check train accuracy
                for data in train_dataloader:
                    if use_cuda:
                        data = {key: value.to(device) for key, value in data.items()}
                    probs = model(data)
                    if is_sequential:
                        mask = data["pad_mask"].view(-1) # [batch_size x max_seq_length] -> [(batch_size*max_seq_length)]
                        macro_accuracy(probs.argmax(-1).view(-1)[mask], data["labels"].view(-1)[mask])
                    else:
                        macro_accuracy(probs.argmax(-1).view(-1), data["labels"].view(-1))
                train_macro_accuracy = macro_accuracy.compute()
                # print(f"Epoch {epoch}: Train_accuracy={total_macro_accuracy}", flush=True)
                macro_accuracy.reset()

                # check valid accuracy
                if args.valid_dataset_path is not None:
                    for data in valid_dataloader:
                        if use_cuda:
                            data = {key: value.to(device) for key, value in data.items()}
                        probs = model(data)
                        if is_sequential:
                            mask = data["pad_mask"].view(-1) # [batch_size x max_seq_length] -> [(batch_size*max_seq_length)]
                            macro_accuracy(probs.argmax(-1).view(-1)[mask], data["labels"].view(-1)[mask])
                        else:
                            macro_accuracy(probs.argmax(-1).view(-1), data["labels"].view(-1))
                    valid_macro_accuracy = macro_accuracy.compute()
                    # print(f"Epoch {epoch}: Valid_accuracy={total_macro_accuracy}", flush=True)
                    macro_accuracy.reset()
            model.train()
            tq1.set_postfix(Train_accuracy=train_macro_accuracy.item(), Valid_accuracy=valid_macro_accuracy.item())
            
            # update the best epoch
            if valid_macro_accuracy >= best_valid_accuracy:
                best_valid_accuracy = valid_macro_accuracy
                with open(f"{args.model_checkpoint_path}/best_epoch.dat", "w") as f:
                    f.write(str(epoch))

            #--------------------
            # update the weights
            #--------------------
            if epoch < args.epochs:
                with tqdm(train_dataloader, leave=False) as tq:
                    tq.set_description(f"Epoch {epoch}")
                    for data in tq:
                        if use_cuda:
                            data = {key: value.to(device) for key, value in data.items()}
                        out = model(data)
                        if is_sequential:
                            loss = loss_func(out, data["labels"], data["pad_mask"])
                        else:
                            loss = loss_func(out, data["labels"])
                        # if is_sequential:
                        #     # mask = data["pad_mask"].view(-1) # [batch_size x max_seq_length] -> [(batch_size*max_seq_length)]
                        #     # # out = out.view(-1, out.size(-1))
                        #     # # bincount = data["labels"].view(-1)[mask].bincount()
                        #     # # weight = bincount.min() / bincount
                        #     # # loss = F.nll_loss(out[mask], data["labels"].view(-1)[mask], weight=weight)
                        #     # bin = batched_bincount(data["labels"].T, 1, out.size(-1)) # [max_seq_length x num_classes]
                        #     # bin_max, _ = bin.max(-1)
                        #     # weight = bin_max[:, None] / (bin + 1e-8) 
                        #     # weight = weight / weight.max(-1, keepdim=True)[0]
                        #     # # weight = (1 - beta) / (1 - beta**bin)
                        #     # # print(weight)
                        #     # loss = 0.0 # torch.FloatTensor([0.0]).to(device)
                        #     # for seq_no in range(weight.size(0)):
                        #     #     loss += F.nll_loss(out[:, seq_no], data["labels"][:, seq_no], weight=weight[seq_no])
                        # else:
                        #     bincount = data["labels"].view(-1).bincount()
                        #     weight = (1 - beta) / (1 - beta**bincount) 
                        #     loss = F.nll_loss(out, data["labels"].squeeze(-1), weight=weight)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        tq.set_postfix(Loss=loss.item())

if __name__ == "__main__":
    import datetime
    import json
    import os
    import argparse
    now = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument("-p", "--problem", default="tsptw", type=str, help="Problem type: [tsptw, cvrptw]")
    parser.add_argument("--gpu", default=-1, type=int, help="Used GPU Number: gpu=-1 indicates using cpu")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers in dataloader")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed for reproductivity")
    # data setting
    parser.add_argument("-train", "--train_dataset_path", type=str, help="Path to a read file", required=True)
    parser.add_argument("-valid", "--valid_dataset_path", type=str, default=None)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--num_cpus", type=int, default=4)
    # training settings
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--cb_beta", default=0.99)
    # parser.add_argument("--valid_interval", default=1, type=int, help="interval outputting intermidiate test accuracy")
    # parser.add_argument("--model_save_interval", type=int, default=1)
    parser.add_argument("--model_checkpoint_path", type=str, default=f"checkpoints/model_{now.strftime('%Y%m%d_%H%M%S')}")
    # model settings
    parser.add_argument("-loss", "--loss_function", type=str, default="seq_cbce", help="[seq_cbce, cbce, wce, ce]")
    parser.add_argument("-node_enc", "--node_enc_type", type=str, default="mlp")
    parser.add_argument("-edge_enc", "--edge_enc_type", type=str, default="attn")
    parser.add_argument("-dec", "--dec_type", type=str, default="lstm")
    parser.add_argument("-pe", "--pos_encoder", type=str, default="sincos")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--num_enc_mlp_layers", type=int, default=2)
    parser.add_argument("--num_dec_mlp_layers", type=int, default=3)
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout probability")
    args = parser.parse_args()

    os.makedirs(args.model_checkpoint_path, exist_ok=True)
    with open(f'{args.model_checkpoint_path}/cmd_args.dat', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    main(args)