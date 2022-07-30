import logging
import warnings
import os
import pickle
import torch
import json
import numpy as np
from .base_dataset import BaseDataset, BaseSetup
import random
UNK = "<unk_token>"
PAD = "<pad_token>"
EMPTY = "<empty>"

logging.basicConfig(level=logging.INFO)



class Setup(BaseSetup):
    def _add_extra_filepaths(self, base_dir):
        if self.use_tree_att:
            self.filepaths["rel_vocab"] = os.path.join(base_dir, "rel_vocab.pkl")
            self.filepaths["tree_rel"] = os.path.join(base_dir, "tree_rel.txt")
            self.filepaths["tree_rel_conv"] = os.path.join(base_dir, f"tree_rel_conv.vocab={self.tree_rel_max_vocab}.txt")


    def _create_vocab(self):
        vocabs = {
            "dp" : TypesValuesVocab(self.filepaths["vocab_types"], self.filepaths["vocab_values"], self.use_anonymized, self.anon_vocab, self.max_values_vocab),
            "tree_rel" : TreeRelVocab(self.filepaths["rel_vocab"], self.tree_rel_max_vocab) if self.use_tree_att else None
        }
        return vocabs

    def _create_dataset(self, fp):
        file_paths = {"dps": fp}
        if self.use_tree_att:
            file_paths.update({"rel_mask" : self.filepaths["tree_rel_conv"]})
        return Dataset(file_paths)

class DPVocab(object):
    def __init__(self, vocab_fp, max_values_vocab=100003):
        super().__init__()
        self.unk_token = UNK
        self.pad_token = PAD
        self.empty_token = EMPTY
        self.pad_idx = None
        self.unk_idx = None
        self.empty_idx = None

        if not os.path.exists(vocab_fp):
            raise Exception("Get the vocab from generate_vocab.py")

        with open(vocab_fp, "rb") as fin:
            self.idx2vocab = pickle.load(fin)
        if max_values_vocab >= 0:
            self.idx2vocab = self.idx2vocab[:min(max_values_vocab, len(self.idx2vocab))]
        logging.info("Loaded vocab from: {}".format(vocab_fp))
        self.vocab2idx = {token: i for i, token in enumerate(self.idx2vocab)}
        self.unk_idx = self.vocab2idx[self.unk_token]
        self.pad_idx = self.vocab2idx[self.pad_token]
        self.empty_idx = self.vocab2idx[self.empty_token]
        logging.info("Vocab size: {}".format(len(self.idx2vocab)))

    def convert(self, dp):

        dp_converted = []
        for token in dp:
            if token in self.vocab2idx:
                dp_converted.append(self.vocab2idx[token]) # in vocab
            else:
                dp_converted.append(self.unk_idx)
        return dp_converted

    def __len__(self):
        return len(self.idx2vocab)

class TypesValuesVocab(object):
    def __init__(self, types_fp=None, values_fp=None, max_values_vocab=100003):
        self.values_vocab = DPVocab(values_fp, max_values_vocab=max_values_vocab)
        if types_fp is not None:
            self.types_vocab = DPVocab(types_fp)
            assert self.values_vocab.unk_idx == self.types_vocab.unk_idx
            assert self.values_vocab.pad_idx == self.types_vocab.pad_idx
            assert self.values_vocab.empty_idx == self.types_vocab.empty_idx
        else:
            self.types_vocab = None
    
    @property
    def unk_idx(self): return self.values_vocab.unk_idx
    @property
    def pad_idx(self): return self.values_vocab.pad_idx
    @property
    def empty_idx(self): return self.values_vocab.empty_idx

    def convert(self, dp):
        (types, values), ext = dp
        if self.types_vocab is not None:
            types_converted = self.types_vocab.convert(types)
        else:
            types_converted = []
        values_converted = self.values_vocab.convert(values)
        dp_converted = (types_converted, values_converted)
        return dp_converted, ext

    def __len__(self):
        return len(self.values_vocab)

class TreeRelVocab(object):
    def __init__(self, rel_vocab_fp=None, tree_rel_max_vocab=10405):
        with open(rel_vocab_fp, "rb") as fin:
            self.idx2rel = pickle.load(fin)
            if tree_rel_max_vocab < len(self.idx2rel):
                self.idx2rel = self.idx2rel[:tree_rel_max_vocab] # crop vocab
        logging.info("Loaded rel vocab from: {}".format(rel_vocab_fp))
        self.rel2idx = {token: i for i, token in enumerate(self.idx2rel)}
        self.rel_unk_idx = self.rel2idx[UNK]
        logging.info("Rel vocab sizes: {}".format(len(self.idx2rel)))

    def convert(self, rel_info):
        rel_converted = [
            [
                self.rel2idx[token] if token in self.rel2idx else self.rel_unk_idx
                for token in rel.split()
            ]
            for rel in rel_info
        ]
        return rel_converted

class Dataset(BaseDataset):
    @staticmethod
    def collate(seqs, values_vocab, args):
        pad_idx = values_vocab.pad_idx
        max_len = max(len(dp["dps"][0][1]) for dp in seqs)
        max_len = max(max_len, 2)
        input_types, input_values = [], []
        target_types, target_values = [], []
        extended = []
        rel_mask = torch.zeros((len(seqs), max_len - 1, max_len - 1)).long() if args.use_tree else []

        for i, dp in enumerate(seqs):
            ((types, values), ext) = dp["dps"]
            if len(values) < 2:
                warnings.warn("got len(values) < 2. skip.")
                continue
            assert len(types) == len(values) or len(types) == 0, (types, values) 
            # ids.append(dp["ids"])
            padding = [pad_idx] * (max_len - len(values))
            if not args.only_values:
                assert len(types) == len(values)
                input_types.append(types[:-1] + padding)
                target_types.append(types[1:] + padding)
            input_values.append(values[:-1] + padding)
            if "rel_mask" in dp:
                mask = dp["rel_mask"]
                assert (len(mask) == len(values) - 1), (len(mask), len(values) - 1)
                # tree relative attention
                for j, rel in enumerate(mask):
                    rel_mask[i][j][: len(rel)] = torch.tensor(rel)
        return {
            "input_seq": {
                "types": torch.tensor(input_types), 
                "values": torch.tensor(input_values)
                },
            "target_seq": {
                "types": torch.tensor(target_types), 
                "values": torch.tensor(target_values)
                },
            "extended": torch.tensor(extended),
            "rel_mask": rel_mask,
        }


def move_to_device(batch, device):
    for key in batch:
        if batch[key] is not None and isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
