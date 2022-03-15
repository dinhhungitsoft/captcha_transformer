import enum
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
    
    train_dataset = dataset.CaptchaDataset(train_imgs, train_targets, resize=(IMAGE_HEIGHT, IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_dataset = dataset.CaptchaDataset(test_imgs, test_targets, test_orig_targets, resize=(IMAGE_HEIGHT, IMAGE_WIDTH))
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

def predict(img, model, lbl_encoder, device = "cpu", max_len = 5, label=""):
    
    model.eval()
    with torch.no_grad():
        encoded,_ = model(img, None, trg_mask = None, is_inference=True)        
    
    
    L = [lbl_encoder.transform(["<"])[0]]   
    
    predicts = [lbl_encoder.transform(["<"])[0]]

    result = []

    for i in range(5):
        with torch.no_grad():         
            for k in range(6 - len(L)):
                L.append(0)   
            trg_tensor = torch.LongTensor(L).unsqueeze(0).to(device)
            trg_mask = model.make_src_mask(trg_tensor)
            output, _ = model.decoder( trg_tensor, encoded,trg_mask = trg_mask, src_mask=None)
            predicted = output.argmax(2)
            idx = L.index(0) - 1
            predicted = predicted[:,idx].item()

            predicts.append(predicted)
            L = predicts.copy()
            
            abc = lbl_encoder.classes_[predicted]
            result.append(abc)
            eos = lbl_encoder.transform([">"])[0]
            if predicted == eos:
                break
    return "".join(result)




train_loader, test_loader, lbl_encoder, test_orig_targets = prepare_data()
model = build_model("/home/hung/learn/pytorch/captcha_trainsformer/weights/model_49.pt")

# data = next(iter(test_loader))


for bindex, data in enumerate(test_loader):
    for j in range(8):
        img = data["images"][j].unsqueeze(0)
        lbl = data["raw_targets"][j]
        abc = predict(img, model, lbl_encoder, "cpu", 5, lbl)
        print(f"{abc}, {lbl}")
#     if isinstance(sentence, str):
#         nlp = spacy.load('de_core_news_sm')
#         tokens = [token.text.lower() for token in nlp(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]

#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
#     src_indexes = [src_field.vocab.stoi[token] for token in tokens]

#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
#     src_mask = model.make_src_mask(src_tensor)
    
#     with torch.no_grad():
#         enc_src = model.encoder(src_tensor, src_mask)

#     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

#     for i in range(max_len):

#         trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

#         trg_mask = model.make_trg_mask(trg_tensor)
        
#         with torch.no_grad():
#             output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
#         pred_token = output.argmax(2)[:,-1].item()
        
#         trg_indexes.append(pred_token)

#         if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
#             break
    
#     trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
#     return trg_tokens[1:], attention

# def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
#     assert n_rows * n_cols == n_heads
    
#     fig = plt.figure(figsize=(15,25))
    
#     for i in range(n_heads):
        
#         ax = fig.add_subplot(n_rows, n_cols, i+1)
        
#         _attention = attention.squeeze(0)[i].cpu().detach().numpy()

#         cax = ax.matshow(_attention, cmap='bone')

#         ax.tick_params(labelsize=12)
#         ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
#                            rotation=45)
#         ax.set_yticklabels(['']+translation)

#         ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()
#     plt.close()