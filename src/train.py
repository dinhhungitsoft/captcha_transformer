import os
import glob
import torch
import numpy as np
from sklearn import model_selection

import config
import dataset
import train_core
from model import CaptchaModel
import utils
from pprint import pprint

def prepare_data():    
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.png"))

    # Get labels
    labels = [x.split("/")[-1][:-4] for x in image_files]    
    
    lbl_encoder = utils.get_lbl_encoder()
   
    targets_enc = [lbl_encoder.transform([c for c in item]) for item in labels]
    targets_enc = np.array(targets_enc) + 1

    # print(targets)    

    (train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets) = model_selection.train_test_split(
        image_files, targets_enc, labels, test_size=0.1, random_state=42)
    
    train_dataset = dataset.CaptchaDataset(train_imgs, train_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    test_dataset = dataset.CaptchaDataset(test_imgs, test_targets, resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, test_loader, lbl_encoder, test_orig_targets

def run_training(from_epoch = 0, weight_path=None):    
    train_loader, test_loader, lbl_encoder, test_orig_targets = prepare_data()

    model = CaptchaModel()

    if weight_path is not None:
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict)

    model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    
    all_epochs = range(config.EPOCH)[from_epoch:]
    for epoch in all_epochs:
        train_loss = train_core.train(model, train_loader, optimizer)
        valid_preds, valid_loss = train_core.eval(model, test_loader)
        
        preds=[]
        for vp in valid_preds:
            current = utils.decode_prediction(vp, lbl_encoder, keep_raw=False)
            preds.extend(current)
        pprint(list(zip(test_orig_targets, preds))[:6])
        print(f"Epoch {epoch}, Train loss: {train_loss}, Val loss: {valid_loss}")
        
        scheduler.step(valid_loss)
        torch.save(model.state_dict(), f"/home/hung/learn/pytorch/ctc/weights/model_{epoch}.pth")


if __name__ == "__main__":
    run_training(from_epoch = 0,weight_path=None)
