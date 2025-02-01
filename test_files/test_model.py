import sys
sys.path.append(".")
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
import torch
import torch.nn as nn
from config.config import config, update_config
from torchinfo import summary
from utils import train_utils

device = "cuda" if torch.cuda.is_available() else "cpu"
update_config("./config/train.yaml")
model = ClipMatcher(config).to(device)
optimizer = train_utils.get_optimizer(config, model)    # TODO (complete): only add the trainable parameters in 

print(train_utils.store_trainable(model, config).keys())
param_size = 0
additional_param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
    if param.requires_grad:
        additional_param_size += param.nelement() * param.element_size()
print("total param size:",param_size)
print("additional param size:",additional_param_size)

# with torch.no_grad():
#     out = model(torch.rand((2,30,3,448,448)).to(device),torch.rand((2,3,448,448)).to(device))
# print(out)
summary(model, [(2,30,3,448,448),(2,3,448,448)])
# print(out["exist_prob"])