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

data_path = os.path.join("./data")
log_path = os.path.join("./log")

_layer_num = 10
_run_num = 2
_epoch_num = 10
_batch_size = 32
_stop_lr = 5e-8
device_id = "cuda:0"
def evaluate_cls(test_data_loader, bert, loss_fct):
    bert.eval()
    loss = 0
    acc = 0
    for batch in tqdm(test_data_loader, desc="[Evaluate]"):
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(device_id),
            text_token_type_ids.to(device_id),
            text_attention_mask.to(device_id),
            label.to(device_id),
        )

        with th.no_grad():
            logits = bert(
                text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=text_token_type_ids,
            )
            

        l = loss_fct(logits.view(-1, 2), label.view(-1))
        loss += l.item()
        acc += (logits.argmax(-1) == label).sum().item()

    return l / len(test_data_loader.dataset), acc / len(test_data_loader.dataset)

def train_cls(
    train_data_loader,
    bert,
    loss_fct,
    optimizer,
    dev_data_loader=None,
    scheduler=None,
):
    # Train the probe
    # probe.train()
    bert.train()
    train_loss, dev_loss = 0, 0
    train_acc, dev_acc = 0, 0
    for batch in tqdm(train_data_loader, desc="[Train]"):
        optimizer.zero_grad()
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(device_id),
            text_token_type_ids.to(device_id),
            text_attention_mask.to(device_id),
            label.to(device_id),
        )

        # with th.no_grad():
        logits = bert(
            text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
        )
        

        l = loss_fct(logits.view(-1, 2), label.view(-1))
        train_loss += l.item()
        l.backward()
        optimizer.step()

        train_acc += (logits.argmax(-1) == label).sum().item()
    train_loss = train_loss / len(train_data_loader.dataset)
    train_acc = train_acc / len(train_data_loader.dataset)

    if dev_data_loader is not None:
        bert.eval()
        for batch in tqdm(dev_data_loader, desc="[Dev]"):
            text_input_ids, text_token_type_ids, text_attention_mask, label = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
            )
            text_input_ids, text_token_type_ids, text_attention_mask, label = (
                text_input_ids.to(device_id),
                text_token_type_ids.to(device_id),
                text_attention_mask.to(device_id),
                label.to(device_id),
            )
            with th.no_grad():
                logits = bert(
                    text_input_ids,
                    attention_mask=text_attention_mask,
                    token_type_ids=text_token_type_ids,
                )
                
                l = loss_fct(logits.view(-1, 2), label.view(-1))
                dev_loss += l.item()

            dev_acc += (logits.argmax(-1) == label).sum().item()
        # Adjust the learning rate
        if scheduler is not None:
            scheduler.step(dev_loss)

        dev_loss = dev_loss / len(dev_data_loader.dataset)
        dev_acc = dev_acc / len(dev_data_loader.dataset)

    return (
        train_loss,
        train_acc,
        dev_loss,
        dev_acc,
    )
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
    argp.add_argument("--dataset",type=str,default="ptb")
    args = argp.parse_args()
    bert_pretrained_file = args.bert_path

    if args.dataset == "ptb":
        args.num_classes = 2
    elif args.dataset == "snli":
        args.num_classes = 3
    # if args.cuda is not None:
    #     device_id = args.cuda
    # else:
    #     device_id, _ = get_max_available_gpu()
    # device = th.device("cuda:" + str(device_id) if th.cuda.is_available() else "cpu")
    # if th.cuda.is_available():
    #     print(f"Using GPU: {device_id}")
    # else:
    #     print("Using CPU")

    timestr = time.strftime("%m%d-%H%M%S")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    train_dataset = th.load(os.path.join(data_path, "train_datasetc.pt"))
    dev_dataset = th.load(os.path.join(data_path, "dev_datasetc.pt"))
    test_dataset = th.load(os.path.join(data_path, "test_datasetc.pt"))

    train_data_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_dataset, batch_size=_batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)
    # bert = BertModel.from_pretrained(bert_pretrained_file)

    model = bertEncoder(args)
    # we are not fine-tuning BERT
    # for param in bert.parameters():
    #     param.requires_grad = False
    model.to(device_id)

    log_file = os.path.join(
        log_path, "layer-" + str(_layer_num) + "-" + timestr + ".log"
    )
    t_total = len(train_data_loader) //  _epoch_num
    avg_acc = []
    for run in tqdm(range(_run_num), desc="[Run]"):
        # probe = PoincareProbe(
        #     device=device, default_dtype=default_dtype, layer_num=_layer_num,
        # )
        # probe.to(device)

        loss_fct = nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=t_total)
        # optimizer = gt.optim.RiemannianAdam(
        #     [
        #         {"params": probe.proj},
        #         {"params": probe.trans},
        #         {"params": probe.pos},
        #         {"params": probe.neg},
        #     ],
        #     lr=1e-3,
        # )
        # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=0)

        with open(log_file, "a") as f:
            f.write(f"Run: {run + 1}\n")
        for epoch in tqdm(range(_epoch_num), desc="[Epoch]"):

            start_time = time.time()
            # train_loss, train_acc, dev_loss, dev_acc = train(
            #     train_data_loader,
            #     probe,
            #     bert,
            #     loss_fct,
            #     optimizer,
            #     dev_data_loader=dev_data_loader,
            #     scheduler=scheduler,
            # )
            train_loss, train_acc, dev_loss, dev_acc = train_cls(
                train_data_loader,
                model,
                loss_fct,
                optimizer,
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            if optimizer.param_groups[0]["lr"] < _stop_lr or epoch == _epoch_num - 1:
                test_loss, test_acc = evaluate_cls(test_data_loader, model, loss_fct)

                # with open(log_file, "a") as f:
                print(
                    f"\nEpoch: {epoch + 1} | time in {mins:.0f} minutes, {secs:.0f} seconds\n"
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

                # break

        avg_acc.append(test_acc)
        if args.save:
            # probe_ckeckpoint = os.path.join(
            #     log_path,
            #     "layer-" + str(_layer_num) + "-run-" + str(run) + "-" + timestr + ".pt",
            # )
            bert_checkpoint = os.path.join(
                log_path,
                "bert-encoder.pt"
            )
            th.save(model.state_dict(), bert_checkpoint)

    # with open(log_file, "a") as f:
    print(f"Avg Acc: {np.mean(avg_acc)*100:.2f}%\n")
