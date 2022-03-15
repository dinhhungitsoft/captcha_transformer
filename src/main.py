import torch
import torch.nn as nn
import numpy as np
import os
from encoderlayers import Encoder
from decoderlayers import Decoder
from model import Seq2Seq, initialize_weights, train, evaluate, epoch_time
from config import *
import glob
from sklearn import preprocessing, model_selection
import dataset
import time
import math

def prepare_data():    
    image_files = glob.glob(os.path.join(DATA_DIR, "*.png"))

    # # Get labels
    # labels = [x.split("/")[-1][:-4] for x in image_files]    
    
    # lbl_encoder = utils.get_lbl_encoder()
   
    # targets_enc = [lbl_encoder.transform([c for c in item]) for item in labels]
    # abc = lbl_encoder.inverse_transform(np.array([49,  4,  6,  9,  9])-1)
    # targets_enc = np.array(targets_enc)
    # targets_enc = targets_enc + 1

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
    
    train_dataset = dataset.CaptchaDataset(train_imgs, train_targets, resize=(IMAGE_WIDTH, IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_dataset = dataset.CaptchaDataset(test_imgs, test_targets, resize=(IMAGE_WIDTH, IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, test_loader, lbl_encoder, test_orig_targets

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

train_loader, test_loader, lbl_encoder, test_orig_targets = prepare_data()


model = build_model("/home/hung/learn/pytorch/captcha_trainsformer/weights/model_265.pt")

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
criterion = nn.CrossEntropyLoss()

best_valid_loss = float('inf')
from_epoch = 265
all_epochs = range(N_EPOCHS)[from_epoch:]
for epoch in all_epochs:
    
    start_time = time.time()
    
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, test_loader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    torch.save(model.state_dict(), f'/home/hung/learn/pytorch/captcha_trainsformer/weights/model_{epoch}.pt')
    scheduler.step(valid_loss)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')