import torch as th

import numpy as np
from tqdm import tqdm

type_num = {"ptb": 2, "snli": 3, "trec": 6, "yelp" : 2}


def evaluate(test_data_loader, probe, bert, loss_fct):
    probe.eval()
    loss = 0
    acc = 0
    for batch in tqdm(test_data_loader, desc="[Evaluate]"):
        text_input_ids, text_attention_mask, text_token_type_ids, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(probe.device),
            text_token_type_ids.to(probe.device),
            text_attention_mask.to(probe.device),
            label.to(probe.device),
        )

        with th.no_grad():
            outputs = bert(
                text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=text_token_type_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs[2]
            sequence_output = (
                hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )
            logits = probe(sequence_output)
        C = type_num["ptb"]
        l = loss_fct(logits.view(-1, C), label.view(-1))
        loss += l.item()
        acc += (logits.argmax(-1) == label).sum().item()

    return l / len(test_data_loader.dataset), acc / len(test_data_loader.dataset)
