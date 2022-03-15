import torch
import torch.nn as nn
import numpy as np
import os
from encoderlayers import Encoder
from decoderlayers import Decoder
from model import train, evaluate, epoch_time
from config import *
import glob
import time
import math
import utils

train_loader, test_loader, lbl_encoder, test_orig_targets = utils.prepare_data()
from_epoch = 47
path=f"/home/hung/learn/pytorch/captcha_trainsformer/weights/model_{from_epoch}.pt"
# path=None
model = utils.build_model(path)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
criterion = nn.CrossEntropyLoss()

best_valid_loss = float('inf')

all_epochs = range(N_EPOCHS)[from_epoch:]
for epoch in all_epochs:
    
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_loader, criterion, lbl_encoder)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(model.state_dict(), f'/home/hung/learn/pytorch/captcha_trainsformer/weights/model_{epoch}.pt')
    scheduler.step(valid_loss)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')