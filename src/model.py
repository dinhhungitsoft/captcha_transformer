import torch.nn as nn
import torch
from tqdm import tqdm
from config import *
from inference import predict
from pprint import pprint
import torch.nn.functional as F


def train(model, data_loader, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    step = 0
    for data in tk:
        for k, v in data.items():        
            data[k] = v.to(DEVICE)
        
        optimizer.zero_grad()
        src = data["images"]        
        trg = data["targets"]      

        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, loss:{loss.item()}")            
        epoch_loss += loss.item()
        step+=1
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion, lbl_encoder=None):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        step = 0
        for data in tk:        
            for k, v in data.items(): 
                if k == "raw_targets":
                    continue
                data[k] = v.to(DEVICE)
                
            src = data["images"]        
            trg = data["targets"]  

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)
            # print(f"val loss {loss.item()}")
            epoch_loss += loss.item()

            for j in range(4):
                img = src[j].unsqueeze(0)
                lbl = data["raw_targets"][j]
                lbl = lbl.replace("<", "")
                lbl = lbl.replace(">", "")
                res = predict(img, model, lbl_encoder, "cpu", 5, lbl)
                pprint(f"{res}, {lbl}")
        
    return epoch_loss / len(data_loader)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

