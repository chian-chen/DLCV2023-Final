import torch
import torch.nn as nn
from math import log2
from utils.model_utils import positionalencoding1d

# similar to spatial-temporal transformer layer
# but maybe less layer or less hidden dim
# To predict if the query object is exist
# Goal: improve tAP@.25
class ExistPredHead(nn.Module):
    def __init__(self, input_size, output_size=1, side_len=8, config=None, from_spatial=False, temporal=False):
        super(ExistPredHead, self).__init__()
        # parameters
        self.input_size = input_size
        self.output_size = output_size
        self.side_len = side_len
        
        self.from_spatial = from_spatial
        # if from_spatial:
        #     # transformer
        #     hidden_size = config["hidden_size"]
        #     num_heads = config["num_heads"]
        #     num_layers = config["num_layers"]
        #     self.embedding = nn.Embedding(input_size, hidden_size)
        #     self.transformer_encoder = nn.TransformerEncoder(
        #         nn.TransformerEncoderLayer(
        #             d_model=input_size, 
        #             nhead=num_heads,
        #             dim_feedforward=hidden_size,
        #             dropout=0.2,
        #             activation="gelu",
        #             batch_first=True
        #             ),
        #             num_layers=num_layers
        #     )
            
        # Downsampling, by Conv2d or MaxPooling2d
        self.num_head_layers, self.down_heads = int(log2(side_len)), []

        for i in range(self.num_head_layers):   
            self.in_channel = 256 if i != 0 else input_size
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        
        self.down_heads = nn.Sequential(*self.down_heads)

        # add transformer
        self.temporal = temporal
        if temporal:
            self.temporal_mask = None
            self.temporal_transformer =  []
            for _ in range(config["num_layers"]):
                
                self.temporal_transformer.append(
                    nn.TransformerEncoderLayer(
                    d_model=input_size,
                    nhead=config["num_heads"],
                    dim_feedforward=config["hidden_size"],
                    dropout=0.2,
                    activation="gelu",
                    batch_first=True
                    ),
                )
            self.temporal_transformer = nn.ModuleList(self.temporal_transformer)
            self.pe = positionalencoding1d(d_model=input_size, length=config["clip_num_frames"])
            self.pe = nn.parameter.Parameter(self.pe)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x = self.embedding(x)
        # x = x.permute(1, 0, 2)  # Change from (batch_size, sequence_length, hidden_size) to (sequence_length, batch_size, hidden_size)
        b, t, h, w, c = x.shape

        output = x.reshape(b*t, h, w, c).permute(0,3,1,2)   # (b*t,8,8,256) 
        output = self.down_heads(output)    # (b*t,1,1,256)
        
        if self.temporal:
            output = output.reshape(b,t,-1) + self.pe
            mask = self.get_mask(output, t)
            for layer in self.temporal_transformer:
                output = layer(output, src_mask=mask)
            output = output.reshape(b*t,-1)

        output = torch.squeeze(output)      # (b*t,256) 
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
            
        output = self.fc(output)        # (b*t,1)
        return output.reshape(b,t)
        
    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            # src: (b,t, 1, 1, c)

            mask = torch.ones(t,t).float() * float('-inf')

            # window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, i-1)
                max_idx = min(t, i+2)
                mask[i: (i+1), min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask

if __name__ == "__main__":
    from torchinfo import summary
    model = ExistPredHead(input_size=256, hidden_size=512, output_size=1, num_heads=16, num_layers=2)
    summary(model, (1, 5, 5, 256))
    # x = model(torch.rand(10,5,5, 256))
    