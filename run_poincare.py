import os
import sys
import pickle
import math
import time
from argparse import ArgumentParser

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils as tutils
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
# from transformers.configuration_bert import BertConfig
import geoopt as gt

import numpy as np
from tqdm import tqdm

from probe.probe import *
from util.train import train
from util.evalu import evaluate
from util.cuda import get_max_available_gpu
from encoder.bert_encoder1 import *

default_dtype = th.float64
th.set_default_dtype(default_dtype)

log_path = os.path.join("./log")

_layer_num = 11
_run_num = 5
_epoch_num = 40
_batch_size = 32
_stop_lr = 5e-8

if __name__ == "__main__":
    """
    config
    """
    argp = ArgumentParser()
    argp.add_argument("--save", type=bool, default=True, help="Save probe")
    argp.add_argument("--cuda", type=int, help="CUDA device")
    argp.add_argument("--bert_path",type=str,default="")
    argp.add_argument("--num_classes",default=2,type=int,help="number of classes")
    argp.add_argument("--hidden_dropout_prob",type=float,default=0.1)
    argp.add_argument("--hidden_size",type=int,default=768)
    argp.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    argp.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    argp.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
    argp.add_argument("--poincare_type",type=str, default="snli")

    args = argp.parse_args()
    bert_pretrained_file = args.bert_path


    if args.cuda is not None:
        device_id = args.cuda
    else:
        device_id, _ = get_max_available_gpu()
    # device = th.device("cuda:" + str(device_id) if th.cuda.is_available() else "cpu")
    device = "cuda:0"
    if th.cuda.is_available():
        print(f"Using GPU: {device_id}")
    else:
        print("Using CPU")

    timestr = time.strftime("%m%d-%H%M%S")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    data_path = f"./data/{args.poincare_type}"

    train_dataset = th.load(os.path.join(data_path, "train_dataset.pt"))
    dev_dataset = th.load(os.path.join(data_path, "dev_dataset.pt"))
    test_dataset = th.load(os.path.join(data_path, "test_dataset.pt"))

    train_data_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_dataset, batch_size=_batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

    # bert = bertEncoder(args)
    bert = BertModel.from_pretrained(bert_pretrained_file)
    # ckpt = "./checkpoint/bestmodel.pt"
    # with open(ckpt,"rb") as f:
    #     model_state_dict = torch.load(f)
    #     bert.load_state_dict(model_state_dict)
    # we are not fine-tuning BERT
    for param in bert.parameters():
        param.requires_grad = False
    bert.to(device)

    log_file = os.path.join(
        log_path, "layer-" + str(_layer_num) + "-" + timestr + ".log"
    )
    avg_acc = []
    for run in tqdm(range(_run_num), desc="[Run]"):
        probe = PoincareProbe(
            device=device, default_dtype=default_dtype, layer_num=_layer_num, type = args.poincare_type
        )
        probe.to(device)

        loss_fct = nn.CrossEntropyLoss()
        if args.poincare_type == "ptb" or args.poincare_type == "yelp":
            optimizer = gt.optim.RiemannianAdam(
                [
                    {"params": probe.proj},
                    {"params": probe.trans},
                    {"params": probe.pos},
                    {"params": probe.neg},
                ],
                lr=1e-3,
            )
        elif args.poincare_type == "snli":
            optimizer = gt.optim.RiemannianAdam(
                [
                    {"params": probe.proj},
                    {"params": probe.trans},
                    {"params": probe.c1},
                    {"params": probe.c2},
                    {"params": probe.c3}
                ],
                lr = 1e-3
            )
        elif args.poincare_type == "trec":
            optimizer = gt.optim.RiemannianAdam(
                [
                    {"params": probe.proj},
                    {"params": probe.trans},
                    {"params": probe.centriods[0]},
                    {"params": probe.centriods[1]},
                    {"params": probe.centriods[2]},
                    {"params": probe.centriods[3]},
                    {"params": probe.centriods[4]},
                    {"params": probe.centriods[5]},
                ],
                lr = 1e-3
            )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=0)

        # with open(log_file, "a") as f:
        #     f.write(f"Run: {run + 1}\n")
        print(f"Run: {run + 1}\n")
        for epoch in tqdm(range(_epoch_num), desc="[Epoch]"):

            start_time = time.time()
            train_loss, train_acc, dev_loss, dev_acc = train(
                train_data_loader,
                probe,
                bert,
                loss_fct,
                optimizer,
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            if optimizer.param_groups[0]["lr"] < _stop_lr or epoch == _epoch_num - 1:
                test_loss, test_acc = evaluate(test_data_loader, probe, bert, loss_fct)
               
                
                print(
                    f"Epoch: {epoch + 1} | time in {mins:.0f} minutes, {secs:.0f} seconds\n"
                )
                print(
                    f"\tTrain Loss: {train_loss:.4f}\t|\tTrain Acc: {train_acc * 100:.2f}%\n"
                )
                print(
                    f"\tDev Loss: {dev_loss:.4f}\t|\tDev Acc: {dev_acc * 100:.2f}%\n"
                )
                print(
                    f"\tTest Loss:  {test_loss:.4f}\t|\tTest Acc:  {test_acc * 100:.2f}%\n"
                )
                print("-" * 50 + "\n")

                break

        avg_acc.append(test_acc)
        # if args.save:
        # if 
        probe_ckeckpoint = f"./checkpoint/{args.poincare_type}/layer=11_dim=64.pt"
        th.save(probe.state_dict(), probe_ckeckpoint)

    # with open(log_file, "a") as f:
    #     f.write(f"Avg Acc: {np.mean(avg_acc)*100:.2f}%\n")
    print(f"Avg Acc: {np.mean(avg_acc)*100:.2f}%\n")
