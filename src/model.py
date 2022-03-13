import config
from logging import captureWarnings
from unicodedata import bidirectional
from pyparsing import cpp_style_comment
import torch
from torch import dropout, nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()        

        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(768, 64)
        self.drop_1 = nn.Dropout(0.2)
        
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        self.output = nn.Linear(64, config.NUM_CHARS + 1)
    
    def forward(self, images, targets=None):
        bs, c, h, w = images.size()
        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool_2(x)
        # print(x.size())

        x = x.permute(0, 3, 1, 2)                
        # print(x.size())
        x = x.view(bs, x.size(1), -1)
        # print(x.size())
        x = self.linear_1(x)
        x = self.drop_1(x)        
        # print(x.size())

        x, _ = self.gru(x)
        # print(x.size())
        x = self.output(x)
        # print(x.size())

        x = x.permute(1, 0, 2)
        if targets is not None:
            log_softmax_vals = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), 
                fill_value=log_softmax_vals.size(0),
                dtype = torch.int32
            )
            # print(f"input_lengths {input_lengths}")
            
            target_lengths = torch.full(
                size=(bs,), 
                fill_value=targets.size(1),
                dtype = torch.int32
            )            
            # print(f"target_lengths {target_lengths}")
            
            loss = nn.CTCLoss(blank=0)(
                log_softmax_vals,
                targets,
                input_lengths,
                target_lengths
            )
            return x, loss
            
        return x, None

if __name__ == "__main__":
    model = CaptchaModel()    
    image = torch.rand((1, 3, 75,300))
    out = model(image, torch.randint(0, 1, (1, 5)))