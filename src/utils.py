import spacy
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import numpy as np
import os
from encoderlayers import Encoder
from decoderlayers import Decoder
from config import *
import glob
from sklearn import preprocessing, model_selection
import dataset
import time
import math

def prepare_data():    
    image_files = glob.glob(os.path.join(DATA_DIR, "*.png"))
   
    labels = [x.split("/")[-1][:-4] for x in image_files]
    labels = [f"<{l}>" for l in labels]
    
    targets = [[c for c in x] for x in labels]
    targets_flat = [c for clist in targets for c in clist]
    
    targets_flat.append(' ')

    lbl_encoder = preprocessing.LabelEncoder()
    lbl_encoder.fit(targets_flat)
    targets_enc = [lbl_encoder.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)     

    sos_token_idx = lbl_encoder.transform(["<"])
    eos_token_idx = lbl_encoder.transform([">"])
    blank = lbl_encoder.transform([" "])
    vocab_size = lbl_encoder.classes_

    (train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets) = model_selection.train_test_split(
        image_files, targets_enc, labels, test_size=0.1, random_state=42)
    
    train_dataset = dataset.CaptchaDataset(train_imgs, train_targets, resize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_dataset = dataset.CaptchaDataset(test_imgs, test_targets, test_orig_targets, resize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader, lbl_encoder, test_orig_targets

def display_attention(predicted, attention, n_heads = 4, n_rows = 2, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(5,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=10)
        # ax.set_xticklabels([" " for i in range(10)], rotation=45)
        ax.set_yticklabels([s for s in predicted], rotation=0)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()
    plt.close()