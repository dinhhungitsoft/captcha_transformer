import torch
def padding_mask(input, pad_idx):
    """
    input: (batch_length, seq_length, 1)
    """
    tensor = (input != pad_idx).unsqueeze(1).unsqueeze(2)
    return tensor

def look_ahead_mask(input, device="cpu"):        
    #input = [batch size, trg len, 1]            
    input_len = input.shape[1]    
    mask = torch.tril(torch.ones((input_len, input_len), device = device)).bool()        
    return mask