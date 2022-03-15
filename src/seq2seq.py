import torch.nn as nn
import torch
from config import *
from PatchEmbedding import PatchEmbedding

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
        # self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        # self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        # self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))                
        # self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        # self.layer_norm1 = nn.LayerNorm(64)

        # self.linear_1 = nn.Linear(3200, embedding_size)
        self.patches = PatchEmbedding(emb_size=embedding_size, img_size=img_size)
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
        x = self.patches(src)
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
