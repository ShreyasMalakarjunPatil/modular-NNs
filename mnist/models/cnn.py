import torch
import torch.nn as nn
from models.layers.layers import MaskedConv2d, MaskedLinear

class cnn(nn.Module):
    def __init__(self, mlp_arch = [512,6]):
        super(cnn, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        conv_layers = []
        conv_layers += [MaskedConv2d(2, 12, 5), nn.ReLU(inplace=True)]
        conv_layers += [nn.MaxPool2d(2)]
        conv_layers += [MaskedConv2d(12, 32, 5), nn.ReLU(inplace=True)]
        conv_layers += [nn.MaxPool2d(2)]
        self.conv_layers = nn.Sequential(*conv_layers)
        
        layers = []
        for i in range(1, len(mlp_arch)-1):
            linear = MaskedLinear(mlp_arch[i-1], mlp_arch[i])
            layers += [linear, nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

        self.output_heads = MaskedLinear(mlp_arch[-2], mlp_arch[-1])
        self.output = nn.Sigmoid()
    
    def forward(self, x, train=True):

        x = self.conv_layers(x)

        x = x.view(x.shape[0], -1)
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