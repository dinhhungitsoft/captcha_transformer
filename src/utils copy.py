import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


import numpy as np

import random
import math
import time
import spacy

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 1

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_de, 
        init_token = '<sos>', 
        eos_token = '<eos>', 
        lower = True, 
        batch_first = True)

TRG = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

def prepare_data():
    print("Prepare data..")
    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), 
                                                    fields = (SRC, TRG))    
    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        device = device)
    
    print("Finish preparing data..")
    return train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_iterator, valid_iterator, test_iterator, train_data, valid_data, test_data = prepare_data()