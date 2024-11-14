import torch
import torch.nn as nn
import numpy as np
from models.layers.layers import MaskedConv2d, MaskedLinear

class mlp(nn.Module):
    def __init__(self, arch, num_tasks=1, output_dim=2):
        super(mlp, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        layers = []
        if num_tasks>1:
            arch[0] = arch[0]+num_tasks
        for i in range(1, len(arch)-1):
            linear = MaskedLinear(arch[i-1], arch[i])
            layers += [linear, nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

        self.num_tasks = num_tasks
        self.output_heads = MaskedLinear(arch[-2], arch[-1])
        #self.output_heads = nn.ModuleList()
        #for i in range(arch[-1]):
        #    linear = MaskedLinear(arch[-2], self.num_tasks)
        #    self.output_heads.append(linear)

        self.output = nn.Sigmoid()
    
    def forward(self, x, task_id, train=True):
        #x = self.layers(x)
        #for i in range(len(self.output_heads)):
        #    head_out = self.output_heads[i](x)
        #    head_out = (head_out*torch.nn.functional.one_hot(task_id, num_classes=self.num_tasks).to(self.device)).sum(dim=1).view(x.size(0),1)
        #    if i==0:
        #        y = head_out
        #    else:
        #        y = torch.cat((y, head_out), dim=1)
        #x = y
        #x = self.output(x)
        #print(x)
        if self.num_tasks>1:
            x = torch.cat((x, torch.nn.functional.one_hot(task_id, num_classes=self.num_tasks).to(self.device)), dim=1)
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


class mlp2(nn.Module):
    def __init__(self, arch, num_tasks=1, output_dim=2):
        super(mlp2, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        layers = []
        
        if num_tasks>1:
            arch[0] = arch[0]+1
        for i in range(1, len(arch)-1):
            linear = MaskedLinear(arch[i-1], arch[i])
            layers += [linear, nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*layers)

        self.num_tasks = num_tasks
        self.output_heads = MaskedLinear(arch[-2], arch[-1])

        self.output = nn.Sigmoid()
    
    def forward(self, x, task_id, train=True):

        if self.num_tasks>1:
            #print(x.size(), task_id.view((task_id.size()[0],1)).size())
            x = torch.cat((x, task_id.view((task_id.size()[0],1)).to(self.device)), dim=1)#torch.cat((x, torch.nn.functional.one_hot(task_id, num_classes=self.num_tasks).to(self.device)), dim=1)
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