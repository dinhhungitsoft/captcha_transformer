import spacy
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import numpy as np
import os
from seq2seq import Seq2Seq
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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader, lbl_encoder, test_orig_targets


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
def build_model(weight_path=None):
    x = torch.rand((1, 3, 200,200))    
    enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              DEVICE,
              max_length=INPUT_DIM)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                DEVICE,
                max_length=OUTPUT_DIM)

    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0    

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, DEVICE, img_size=IMAGE_WIDTH, embedding_size=HID_DIM).to(torch.device(DEVICE))
    if weight_path is not None:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        model.apply(initialize_weights)
    return model
