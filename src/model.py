import torch.nn as nn
import torch
from encoderlayers import Encoder, EncoderLayer
from decoderlayers import Decoder, DecoderLayer
import utils
from tqdm import tqdm
import config

import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device,
                 img_size=200,
                 embedding_size=64):
        super().__init__()

        # Convolutions for image
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))                
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.layer_norm1 = nn.LayerNorm(64)

        self.linear_1 = nn.Linear(3200, embedding_size)
        # self.drop_1 = nn.Dropout(0.2)
        #        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg, trg_mask=None, is_inference=False):
        bs = src.size(0)
        x = F.relu(self.conv_1(src))        
        x = self.max_pool_1(x)        
        x = F.relu(self.conv_2(x))        
        x = self.max_pool_2(x)        
        
        x = x.permute(0, 3, 1, 2)                        
        x = x.view(bs, x.size(1), -1)        
        x = self.linear_1(x)
        # x = self.layer_norm1(x)
        # x = self.drop_1(x)                
        # x, _ = self.gru(x)        
        # x = self.output(x)        

        # x = x.permute(1, 0, 2)


        src =x




        # img_patches = self.img_embedding(src)
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        # src_mask = self.make_src_mask(img_patches)
        if trg_mask == None and trg is not None:
            trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, None)
        
        #enc_src = [batch size, src len, hid dim]
        if is_inference == False:                
            output, attention = self.decoder(trg, enc_src, trg_mask, None)
        else:
            return enc_src, None
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)



def train(model, data_loader, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    step = 0
    for data in tk:
        for k, v in data.items():        
            data[k] = v.to(config.DEVICE)
        
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
        if step % 2 == 0:
            print(f"Step {step}, loss:{loss.item()}")            
        epoch_loss += loss.item()
        step+=1
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        step = 0
        for data in tk:        
            for k, v in data.items():        
                data[k] = v.to(config.DEVICE)
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
            print(f"val loss {loss.item()}")
            epoch_loss += loss.item()
        
    return epoch_loss / len(data_loader)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

