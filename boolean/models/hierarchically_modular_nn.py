import torch
import torch.nn as nn
import numpy as np
from models.utils import gumbel_sigmoid, gumbel_softmax

class module_mlp(nn.Module):
    def __init__(self, arch):
        super(module_mlp, self).__init__()
        layers = []
        for i in range(1, len(arch)):
            linear = nn.Linear(arch[i-1], arch[i])
            if i<len(arch)-1:
                layers += [linear, nn.ReLU(inplace=True)]
            else:
                layers += [linear]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class hierarchically_modular(nn.Module):
    def __init__(self, num_modules=[3,3], module_arch=[2,12,1], num_tasks=1, input_dim=6, output_dim=2, topk=2, use_gumbel=False, per_sample_routing=False, tau=1.0, known_routing=False, module_inp_indices=None, known_output=False):
        super(hierarchically_modular, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.num_modules = num_modules
        self.module_arch = module_arch
        self.num_tasks = num_tasks
        self.topk = topk

        self.use_gumbel = use_gumbel
        self.tau = tau
        self.per_sample_routing = per_sample_routing
        self.module_inp_indices = module_inp_indices
        self.known_routing = known_routing
        self.known_output = known_output

        self.module_inp_embeddings = nn.ParameterList()
        self.module_mlps = nn.ModuleList()

        for i in range(len(num_modules)):

            if known_routing==True:
                w = torch.zeros(num_tasks, input_dim, num_modules[i]).to(self.device)
                for n in range(self.num_tasks):
                    for p in range(num_modules[i]):
                        w[n, self.module_inp_indices[n][i][p][0], p] = 1.1
                        w[n, self.module_inp_indices[n][i][p][1], p] = 1.0
            else:
                w = torch.randn(num_tasks, input_dim, num_modules[i]).to(self.device)

            self.module_inp_embeddings.append(nn.Parameter(w))
            
            self.module_mlps.append(nn.ModuleList())
            for j in range(num_modules[i]):
                self.module_mlps[i].append(module_mlp(module_arch))
            
            input_dim = num_modules[i]
        
        if known_routing==True or known_output==True:
            w = torch.zeros(num_tasks, input_dim, 1).to(self.device)
            for n in range(num_tasks):
                w[n, self.module_inp_indices[n][-1][0][0], 0] = 1.1
                w[n, self.module_inp_indices[n][-1][0][1], 0] = 1.0
        else:
            w = torch.randn(num_tasks, input_dim, 1).to(self.device)
        self.module_inp_embeddings.append(nn.Parameter(w))

        self.relu = nn.ReLU()
        self.out = nn.Sigmoid()

    def forward(self, x, task_id, train=True):

        for lkj in range(len(self.num_modules)):
            
            if self.per_sample_routing:
                module_input_embeddings = self.module_inp_embeddings[lkj][task_id]
            else:
                module_input_embeddings = self.module_inp_embeddings[lkj]

            if train:
                scores1, scores2 = gumbel_sigmoid(module_input_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
            else:
                scores1, scores2 = gumbel_sigmoid(module_input_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=False, tau = None)

            for i in range(self.num_modules[lkj]):
                
                if self.per_sample_routing:
                    variable_1 = (x*scores1[:,:,i]).sum(dim=1).view(x.size(0),1)
                    variable_2 = (x*scores2[:,:,i]).sum(dim=1).view(x.size(0),1)
                else:
                    variable_1 = (x*scores1[task_id,:,i]).sum(dim=1).view(x.size(0),1)
                    variable_2 = (x*scores2[task_id,:,i]).sum(dim=1).view(x.size(0),1)
                
                module_inp = torch.cat((variable_1,variable_2), dim=1)
                module_out = self.module_mlps[lkj][i](module_inp)
            
                if i==0:
                    y = module_out
                else:
                    y = torch.cat((y,module_out), dim=1)
            x = y
        
        if self.per_sample_routing:
            task_output_embeddings = self.module_inp_embeddings[-1][task_id]
        else:
            task_output_embeddings = self.module_inp_embeddings[-1]

        if train:
            scores1, scores2 = gumbel_sigmoid(task_output_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
        else:
            scores1, scores2 = gumbel_sigmoid(task_output_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=False, tau = None)

        if self.per_sample_routing:
            variable_1 = (x*scores1[:,:,0]).sum(dim=1).view(x.size(0),1)
            variable_2 = (x*scores2[:,:,0]).sum(dim=1).view(x.size(0),1)
        else:
            variable_1 = (x*scores1[task_id,:,0]).sum(dim=1).view(x.size(0),1)
            variable_2 = (x*scores2[task_id,:,0]).sum(dim=1).view(x.size(0),1)
        
        x = torch.cat((variable_1,variable_2), dim=1)
        x = self.out(x)
        return x



