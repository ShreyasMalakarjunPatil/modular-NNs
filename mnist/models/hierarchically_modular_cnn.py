import torch
import torch.nn as nn
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
        x = x.view(x.size(0),-1)
        x = self.layers(x)
        return x

class module_cnn(nn.Module):
    def __init__(self, arch=[128,3]):
        super(module_cnn, self).__init__()

        conv_layers = []
        conv_layers += [nn.Conv2d(1, 6, 5), nn.ReLU(inplace=True)]
        conv_layers += [nn.MaxPool2d(2)]
        conv_layers += [nn.Conv2d(6, 16, 5), nn.ReLU(inplace=True)]
        conv_layers += [nn.MaxPool2d(2)]
        self.conv_layers = nn.Sequential(*conv_layers)

        self.output_heads = nn.Linear(arch[-2], arch[-1])
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.output_heads(x)
        return x

class hierarchically_modular_cnn(nn.Module):
    def __init__(self, num_img_modules=[2], num_modules=[3,2], img_module_arch=[256,3], module_arch=[2,12,1], input_dim=2, output_dim=2, topk=2, use_gumbel=False, per_sample_routing=False, tau=1.0, known_routing=False, module_inp_indices=None):
        super(hierarchically_modular_cnn, self).__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.num_img_modules = num_img_modules
        self.num_modules = num_modules

        self.img_module_arch = img_module_arch
        self.module_arch = module_arch

        self.topk = topk

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.use_gumbel = use_gumbel
        self.tau = tau
        self.per_sample_routing = per_sample_routing
        self.module_inp_indices = module_inp_indices
        self.known_routing = known_routing

        self.module_inp_embeddings = nn.ParameterList()
        self.img_modules = nn.ModuleList()
        self.module_mlps = nn.ModuleList()

        for i in range(len(num_img_modules)):

            if known_routing==True:
                w = torch.zeros(1, input_dim, num_img_modules[i]).to(self.device)
                for p in range(num_img_modules[i]):
                    w[0, self.module_inp_indices[i][p][0], p] = 1.0
            else:
                w = torch.randn(1, input_dim, num_img_modules[i]).to(self.device)

            self.module_inp_embeddings.append(nn.Parameter(w))
            
            self.img_modules.append(nn.ModuleList())
            for j in range(num_img_modules[i]):
                self.img_modules[i].append(module_cnn(img_module_arch))
            
            input_dim = num_img_modules[i]*img_module_arch[-1]
        
        for i in range(len(num_modules)):

            if known_routing==True:
                w = torch.zeros(1, input_dim, num_modules[i]).to(self.device)
                for p in range(num_modules[i]):
                    w[0, self.module_inp_indices[i+len(num_img_modules)][p][0], p] = 1.1
                    w[0, self.module_inp_indices[i+len(num_img_modules)][p][1], p] = 1.0
            else:
                w = torch.randn(1, input_dim, num_modules[i]).to(self.device)
            self.module_inp_embeddings.append(nn.Parameter(w))

            self.module_mlps.append(nn.ModuleList())
            for j in range(num_modules[i]):
                self.module_mlps[i].append(module_mlp(module_arch))

            input_dim = num_modules[i]

        if known_routing==True:
            w = torch.zeros(1, input_dim, 1).to(self.device)
            w[0, self.module_inp_indices[-1][0][0], 0] = 1.1
            w[0, self.module_inp_indices[-1][0][1], 0] = 1.0
        else:
            w = torch.randn(1, input_dim, 1).to(self.device)
        self.module_inp_embeddings.append(nn.Parameter(w))

        self.out = nn.Sigmoid()

    def forward(self, x, train=True):

        for lkj in range(len(self.num_img_modules)):
            
            module_input_embeddings = self.module_inp_embeddings[lkj]

            if train:
                scores = gumbel_softmax(module_input_embeddings, hard = True, dim = 1, use_gumbel=self.use_gumbel, tau = self.tau)
            else:
                scores = gumbel_softmax(module_input_embeddings, hard = True, dim = 1, use_gumbel=False, tau = None)

            for i in range(self.num_img_modules[lkj]):
                
                s = scores[:,:,i]
                s = s.repeat(28,1).repeat(28,1,1).permute(2, 0, 1).unsqueeze(0)
                module_inp = (x*s).sum(dim=1).unsqueeze(1)
                
                module_out = self.img_modules[lkj][i](module_inp)
            
                if i==0:
                    y = module_out
                else:
                    y = torch.cat((y,module_out), dim=1)
            x = y
        
        for lkj in range(len(self.num_modules)):

            module_input_embeddings = self.module_inp_embeddings[lkj+len(self.num_img_modules)]

            if train:
                scores1, scores2 = gumbel_sigmoid(module_input_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
            else:
                scores1, scores2 = gumbel_sigmoid(module_input_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=False, tau = None)

            for i in range(self.num_modules[lkj]):

                variable_1 = (x*scores1[:,:,i]).sum(dim=1).view(x.size(0),1)
                variable_2 = (x*scores2[:,:,i]).sum(dim=1).view(x.size(0),1)

                module_inp = torch.cat((variable_1,variable_2), dim=1)
                module_out = self.module_mlps[lkj][i](module_inp)

                if i==0:
                    y = module_out
                else:
                    y = torch.cat((y,module_out), dim=1)
            
            x = y

        task_output_embeddings = self.module_inp_embeddings[-1]

        if train:
            scores1, scores2 = gumbel_sigmoid(task_output_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
        else:
            scores1, scores2 = gumbel_sigmoid(task_output_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=False, tau = None)

        variable_1 = (x*scores1[:,:,0]).sum(dim=1).view(x.size(0),1)
        variable_2 = (x*scores2[:,:,0]).sum(dim=1).view(x.size(0),1)

        x = torch.cat((variable_1,variable_2), dim=1)
        x = self.out(x)
        return x