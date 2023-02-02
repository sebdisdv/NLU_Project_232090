import Bert.dataloader as bert_dl
import Bert.train as bert_tr
import Baseline.dataloader as baseline_dl
import Baseline.model_GRU as baseline_model_gru
import Baseline.model_LSTM as baseline_model_lstm
import Baseline.train as baseline_tr
import argparse

from transformers import BertConfig
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from datetime import datetime
from log import LogFile
from dataclasses import dataclass

import random
import torch
import numpy


@dataclass
class Settings:
    model: str
    task: str
    crf: bool
    dropout: bool
    runs: int
    device: str

    def __str__(self) -> str:
        return f"model:{self.model}\ntask:{self.task}\ncrf:{self.crf}\ndropout:{self.dropout}\nruns:{self.runs}\ndevice:{self.device}\n"


def main_jointBert(logfile: LogFile, settings: argparse.Namespace):
    dataset_path = f"./IntentSlotDatasets/{settings.task.upper()}"

    PAD_TOKEN = 0  # Specifies a target value that is ignored and does not contribute to the input gradient

    pad_label = "PAD"
    logfile.write(f"pad_label: {pad_label}\n")

    unk_label = "UNK"
    logfile.write(f"unk_label: {unk_label}\n")

    max_seq_len = 50  # The maximum total input sequence length after tokenization.
    logfile.write(f"max_seq_len: {max_seq_len}\n")

    train_batch_size = 64  # Batch size for
    logfile.write(f"train_batch_size: {train_batch_size}\n")

    eval_batch_size = 64  # Batch size for evaluation
    logfile.write(f"eval_batch_size: {eval_batch_size}\n")

    epochs = 3  # Total number of training epochs to perform
    logfile.write(f"epochs: {epochs}\n")

    learning_rate = 5e-5  # The initial learning rate for Adam
    logfile.write(f"learning_rate: {learning_rate}\n")

    adam_epsilon = 1e-8  # Epsilon for Adam optimizer
    logfile.write(f"adam_epsilon: {adam_epsilon}\n")

    max_grad_norm = 1.0  # Max gradient norm
    logfile.write(f"max_grad_norm: {max_grad_norm}\n")

    dropout_rate = 0.1  # Dropout for fully-connected layers
    logfile.write(f"dropout_rate: {dropout_rate}\n")

    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # charge raw data
    train_raw = bert_dl.load_data(dataset_path + "/train.json")
    test_raw = bert_dl.load_data(dataset_path + "/test.json")

    print("Train samples len:", len(train_raw))
    print("Test samples len:", len(test_raw))

    logfile.write(f"Train samples len:{len(train_raw)}\n")
    logfile.write(f"Test samples len:{len(test_raw)}\n")

    # find intent and slot labels
    intent_labels = [unk_label]
    slot_labels = [pad_label, unk_label]
    corpus = train_raw + test_raw

    data_intents = set([line["intent"] for line in corpus])
    data_slots = set(sum([line["slots"].split() for line in corpus], []))

    intent_labels.extend(data_intents)
    slot_labels.extend(data_slots)

    print("len intent labels: ", len(intent_labels))
    logfile.write(f"len intent labels: {len(intent_labels)}\n")

    print("len slot labels: ", len(slot_labels))
    logfile.write(f"len slot labels:  {len(slot_labels)}\n")

    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # create dev raw data
    train_raw, eval_raw, test_raw = bert_dl.create_eval_set(train_raw, test_raw)

    # create datasets
    train_datas = bert_dl.Processed_data(
        train_raw,
        intent_labels,
        slot_labels,
        unk_label,
        pad_label,
        max_seq_len,
        tokenizer,
        PAD_TOKEN,
    )
    eval_datas = bert_dl.Processed_data(
        eval_raw,
        intent_labels,
        slot_labels,
        unk_label,
        pad_label,
        max_seq_len,
        tokenizer,
        PAD_TOKEN,
    )
    test_datas = bert_dl.Processed_data(
        test_raw,
        intent_labels,
        slot_labels,
        unk_label,
        pad_label,
        max_seq_len,
        tokenizer,
        PAD_TOKEN,
    )

    train_dataset = bert_dl.JointBert_Dataset(train_datas)
    eval_dataset = bert_dl.JointBert_Dataset(eval_datas)
    test_dataset = bert_dl.JointBert_Dataset(test_datas)

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False
    )

    # charge pretrained Bert model configurations
    config = BertConfig.from_pretrained(
        "bert-base-uncased", finetuning_task=settings.task
    )

    # do training and evaluation
    if settings.runs <= 1:
        bert_tr.single_run(
            logfile,
            epochs,
            config,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            settings.task,
            dropout_rate,
            learning_rate,
            adam_epsilon,
            max_grad_norm,
            intent_labels,
            slot_labels,
            settings.crf,
            settings.dropout,
            PAD_TOKEN,
            tokenizer,
            settings.device,
        )
    else:
        bert_tr.multiple_runs(
            logfile,
            settings.runs,
            epochs,
            config,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            dropout_rate,
            learning_rate,
            adam_epsilon,
            max_grad_norm,
            intent_labels,
            slot_labels,
            settings.crf,
            settings.dropout,
            PAD_TOKEN,
            tokenizer,
            settings.device,
        )


def main_baseline_GRU(logfile: LogFile, settings: Settings):
    dataset_path = f"./IntentSlotDatasets/{settings.task.upper()}"

    PAD_TOKEN = 0  # Specifies a target value that is ignored and does not contribute to the input gradient

    train_batch_size = 256  # Batch size for training
    logfile.write(f"train_batch_size:{train_batch_size}\n")

    eval_batch_size = 64  # Batch size for evaluation
    logfile.write(f"eval_batch_size:{eval_batch_size}\n")

    hid_size = 200  # size of the hidden layers
    logfile.write(f"hid_size:{hid_size}\n")

    n_layer = 1  # number of hidden layers in the encoder
    logfile.write(f"n_layer:{n_layer}\n")

    emb_size = 300  # size of the embedding
    logfile.write(f"emb_size:{emb_size}\n")

    epochs = 100  # Total number of training epochs to perform
    logfile.write(f"epochs:{epochs}\n")

    patience = 201  # parameter for early stopping
    logfile.write(f"patience:{patience}\n")

    learning_rate = 0.0001  # The initial learning rate for Adam
    logfile.write(f"learning_rate:{learning_rate}\n")

    max_grad_norm = 5  # Max gradient norm
    logfile.write(f"max_grad_norm:{max_grad_norm}\n")

    dropout_rate = 0.1  # Dropout for fully-connected layers
    logfile.write(f"dropout_rate:{dropout_rate}\n")

    train_raw = bert_dl.load_data(dataset_path + "/train.json")
    test_raw = bert_dl.load_data(dataset_path + "/test.json")
    logfile.write(f"Train samples len:{len(train_raw)}\n")
    logfile.write(f"Test samples len:{len(test_raw)}\n")

    # find intent and slot labels
    intent_labels = []
    slot_labels = []
    corpus = train_raw + test_raw
    data_intents = set([line["intent"] for line in corpus])
    data_slots = set(sum([line["slots"].split() for line in corpus], []))
    intent_labels.extend(data_intents)
    slot_labels.extend(data_slots)

    print("len intent labels: ", len(intent_labels))
    logfile.write(f"len intent labels: {len(intent_labels)}\n")

    print("len slot labels: ", len(slot_labels))
    logfile.write(f"len slot labels:  {len(slot_labels)}\n")
    # create dev raw data
    train_raw, eval_raw, test_raw = bert_dl.create_eval_set(train_raw, test_raw)

    # create datasets
    words = sum([x["utterance"].split() for x in train_raw], [])

    lang = baseline_dl.Lang(words, intent_labels, slot_labels, PAD_TOKEN, cutoff=0)

    train_dataset = baseline_dl.IntentsAndSlots(train_raw, lang)
    eval_dataset = baseline_dl.IntentsAndSlots(eval_raw, lang)
    test_dataset = baseline_dl.IntentsAndSlots(test_raw, lang)

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=baseline_dl.collate_fn,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, collate_fn=baseline_dl.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, collate_fn=baseline_dl.collate_fn
    )

    num_slot_labels = len(lang.slot2id)
    num_intent_labels = len(lang.intent2id)

    vocab_len = len(lang.word2id)
    print("vocab len: ", vocab_len)
    logfile.write(f"vocab len: {vocab_len}\n")

    # create model
    model = baseline_model_gru.Base_Model(
        hid_size,
        num_slot_labels,
        num_intent_labels,
        emb_size,
        vocab_len,
        settings.crf,
        settings.dropout,
        dropout_rate,
        n_layer,
        pad_token=PAD_TOKEN,
    )

    print(model)

    if settings.runs <= 1:
        baseline_tr.single_run(
            logfile,
            epochs,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lang,
            settings.task,
            learning_rate,
            max_grad_norm,
            patience,
            intent_labels,
            slot_labels,
            settings.crf,
            settings.dropout,
            PAD_TOKEN,
            settings.device,
        )
    else:
        baseline_tr.multiple_runs(
            logfile,
            settings.runs,
            epochs,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lang,
            learning_rate,
            max_grad_norm,
            patience,
            settings.crf,
            PAD_TOKEN,
            settings.device,
        )


def main_baseline_LSTM(logfile: LogFile, settings: Settings):
    dataset_path = f"./IntentSlotDatasets/{settings.task.upper()}"

    PAD_TOKEN = 0  # Specifies a target value that is ignored and does not contribute to the input gradient

    train_batch_size = 256  # Batch size for training
    logfile.write(f"train_batch_size:{train_batch_size}\n")

    eval_batch_size = 64  # Batch size for evaluation
    logfile.write(f"eval_batch_size:{eval_batch_size}\n")

    hid_size = 200  # size of the hidden layers
    logfile.write(f"hid_size:{hid_size}\n")

    n_layer = 1  # number of hidden layers in the encoder
    logfile.write(f"n_layer:{n_layer}\n")

    emb_size = 300  # size of the embedding
    logfile.write(f"emb_size:{emb_size}\n")

    epochs = 100  # Total number of training epochs to perform
    logfile.write(f"epochs:{epochs}\n")

    patience = 201  # parameter for early stopping
    logfile.write(f"patience:{patience}\n")

    learning_rate = 0.0001  # The initial learning rate for Adam
    logfile.write(f"learning_rate:{learning_rate}\n")

    max_grad_norm = 5  # Max gradient norm
    logfile.write(f"max_grad_norm:{max_grad_norm}\n")

    dropout_rate = 0.1  # Dropout for fully-connected layers
    logfile.write(f"dropout_rate:{dropout_rate}\n")

    # charge raw data
    train_raw = bert_dl.load_data(dataset_path + "/train.json")
    test_raw = bert_dl.load_data(dataset_path + "/test.json")
    logfile.write(f"Train samples len:{len(train_raw)}\n")
    logfile.write(f"Test samples len:{len(test_raw)}\n")

    # find intent and slot labels
    intent_labels = []
    slot_labels = []
    corpus = train_raw + test_raw
    data_intents = set([line["intent"] for line in corpus])
    data_slots = set(sum([line["slots"].split() for line in corpus], []))
    intent_labels.extend(data_intents)
    slot_labels.extend(data_slots)

    print("len intent labels: ", len(intent_labels))
    logfile.write(f"len intent labels: {len(intent_labels)}\n")

    print("len slot labels: ", len(slot_labels))
    logfile.write(f"len slot labels:  {len(slot_labels)}\n")
    # create dev raw data
    train_raw, eval_raw, test_raw = bert_dl.create_eval_set(train_raw, test_raw)

    # create datasets
    words = sum([x["utterance"].split() for x in train_raw], [])

    lang = baseline_dl.Lang(words, intent_labels, slot_labels, PAD_TOKEN, cutoff=0)

    train_dataset = baseline_dl.IntentsAndSlots(train_raw, lang)
    eval_dataset = baseline_dl.IntentsAndSlots(eval_raw, lang)
    test_dataset = baseline_dl.IntentsAndSlots(test_raw, lang)

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=baseline_dl.collate_fn,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, collate_fn=baseline_dl.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=eval_batch_size, collate_fn=baseline_dl.collate_fn
    )

    num_slot_labels = len(lang.slot2id)
    num_intent_labels = len(lang.intent2id)

    vocab_len = len(lang.word2id)
    print("vocab len: ", vocab_len)
    logfile.write(f"vocab len: {vocab_len}\n")

    # create model
    model = baseline_model_lstm.Base_Model(
        hid_size,
        num_slot_labels,
        num_intent_labels,
        emb_size,
        vocab_len,
        settings.crf,
        settings.dropout,
        dropout_rate,
        n_layer,
        pad_token=PAD_TOKEN,
    )

    print(model)

    if settings.runs <= 1:
        baseline_tr.single_run(
            logfile,
            epochs,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lang,
            settings.task,
            learning_rate,
            max_grad_norm,
            patience,
            intent_labels,
            slot_labels,
            settings.crf,
            settings.dropout,
            PAD_TOKEN,
            settings.device,
        )
    else:
        baseline_tr.multiple_runs(
            logfile,
            settings.runs,
            epochs,
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            lang,
            learning_rate,
            max_grad_norm,
            patience,
            settings.crf,
            PAD_TOKEN,
            settings.device,
        )


if __name__ == "__main__":
    # Set seeds to make experiment repeatable
    torch.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        "--model",
        default=None,
        required=True,
        type=str,
        help="Available model ['LSTM', 'GRU', 'JOINTBERT']",
    )
    args_parser.add_argument(
        "--task",
        default=None,
        required=True,
        type=str,
        help="Task available are ['snips', 'atis']",
    )
    args_parser.add_argument(
        "--device",
        default="cpu",
        required=False,
        type=str,
        help="'cuda' to run on gpu if possible",
    )
    args_parser.add_argument(
        "--crf",
        default=0,
        required=True,
        type=int,
        help="1 to Add CRF layer in the slot classifier",
    )
    args_parser.add_argument(
        "--dropout",
        default=0,
        required=True,
        type=int,
        help="1 to use dropout as normalization method, 0 for layer normalization",
    )
    args_parser.add_argument(
        "--runs", default=1, required=True, type=int, help="Number of runs to process"
    )

    args_parsed = args_parser.parse_args()

    logfile = LogFile(name=datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))

    settings = Settings(
        model=args_parsed.model,
        crf=True if args_parsed.crf == 1 else False,
        task=args_parsed.task,
        dropout=True if args_parsed.dropout == 1 else False,
        runs=args_parsed.runs,
        device=args_parsed.device,
    )

    print(str(settings))
    logfile.write(str(settings))

    if settings.model == "LSTM":
        main_baseline_LSTM(logfile, settings)
    elif settings.model == "GRU":
        main_baseline_GRU(logfile, settings)
    elif settings.model == "JOINTBERT":
        main_jointBert(logfile, settings)
