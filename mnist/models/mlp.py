import torch
import torch.nn as nn
from models.layers.layers import MaskedConv2d, MaskedLinear

class mlp(nn.Module):
    def __init__(self, arch):
        super(mlp, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        layers = []

        for i in range(1, len(arch)-1):
            linear = MaskedLinear(arch[i-1], arch[i])
            layers += [linear, nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

        self.output_heads = MaskedLinear(arch[-2], arch[-1])
        self.output = nn.Sigmoid()
    
    def forward(self, x, train=True):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.output_heads(x)
        x = self.output(x)
        return x
   
    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1