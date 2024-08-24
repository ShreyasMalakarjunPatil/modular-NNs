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

class NNmodules(nn.Module):
    def __init__(self, num_modules=1, module_arch=[2,12,1]):
        super(NNmodules, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_modules = num_modules
        self.module_arch = module_arch
        self.mlps = nn.ModuleList()

        for i in range(num_modules):
            self.mlps.append(module_mlp(module_arch))
    
    def forward(self, x):
        for i in range(self.num_modules):
            module_out = self.mlps[i](x)
            if i==0:
                y = module_out
            else:
                y = torch.cat((y,module_out), dim=1)
        return y

class NNmodules_img(nn.Module):
    def __init__(self, num_modules=1, module_arch=[784,256,256,3]):
        super(NNmodules_img, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_modules = num_modules
        self.module_arch = module_arch
        self.mlps = nn.ModuleList()

        for i in range(num_modules):
            self.mlps.append(module_mlp(module_arch))
    
    def forward(self, x, score):
        for i in range(self.num_modules):
            module_out = self.mlps[i](x)
            if i==0:
                y = module_out*score[:,i]
            else:
                y += module_out*score[:,i]
        return y

class hierarchically_modular_shared_modules_mlp(nn.Module):
    def __init__(self, num_img_modules=1, num_modules=2, num_img_slots=[2], num_slots=[3,2], img_module_arch=[784,256,256,3], module_arch=[2,12,1], input_dim=2, output_dim=2, topk=2, use_gumbel=False, per_sample_routing=False, tau=1.0, known_routing=False, known_module_location=False, module_inp_indices=None, module_slot_indices=None):
        super(hierarchically_modular_shared_modules_mlp, self).__init__()
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.num_img_modules = num_img_modules
        self.num_modules = num_modules

        self.num_img_slots = num_img_slots
        self.num_slots = num_slots
        
        self.img_module_arch = img_module_arch
        self.module_arch = module_arch

        self.topk = topk

        self.tau = tau
        self.use_gumbel = use_gumbel
        self.per_sample_routing = per_sample_routing
        self.module_inp_indices = module_inp_indices
        self.module_slot_indices = module_slot_indices
        self.known_routing = known_routing
        self.known_module_location = known_module_location

        self.img_modules = NNmodules_img(self.num_img_modules, self.img_module_arch)
        self.module_mlps = NNmodules(self.num_modules, self.module_arch)

        self.module_loc_embeddings = nn.ParameterList()
        self.module_inp_embeddings = nn.ParameterList()

        for i in range(len(self.num_img_slots)):

            if known_routing==True:
                w = torch.zeros(1, input_dim, num_img_slots[i]).to(self.device)
                for p in range(num_img_slots[i]):
                    w[0, self.module_inp_indices[i][p][0], p] = 1.0
            else:
                w = torch.randn(1, input_dim, num_img_slots[i]).to(self.device)

            self.module_inp_embeddings.append(nn.Parameter(w))
            
            if known_module_location==True:
                w2 = torch.zeros(1, self.num_img_modules, num_img_slots[i]).to(self.device)
                w2[0, self.module_slot_indices[i], torch.arange(0,num_img_slots[i])] = 1.0
            else:
                w2 = torch.randn(1, self.num_img_modules, num_img_slots[i]).to(self.device)

            self.module_loc_embeddings.append(nn.Parameter(w2))

            input_dim = num_img_slots[i]*img_module_arch[-1]

        
        for i in range(len(self.num_slots)):
            
            if known_routing==True:
                w = torch.zeros(1, input_dim, num_slots[i]).to(self.device)
                for p in range(num_slots[i]):
                    w[0, self.module_inp_indices[i+len(num_img_slots)][p][0], p] = 1.1
                    w[0, self.module_inp_indices[i+len(num_img_slots)][p][1], p] = 1.0
            else:
                w = torch.randn(1, input_dim, num_slots[i]).to(self.device)
                
            self.module_inp_embeddings.append(nn.Parameter(w))
                
            if known_module_location==True:
                w2 = torch.zeros(1, self.num_modules, num_slots[i]).to(self.device)
                w2[0, self.module_slot_indices[i+len(num_img_slots)], torch.arange(0,num_slots[i])] = 1.0
            else:
                w2 = torch.randn(1, self.num_modules, num_slots[i]).to(self.device)

            self.module_loc_embeddings.append(nn.Parameter(w2))

            input_dim = num_slots[i]
                    
        if known_routing==True:
            w = torch.zeros(1, input_dim, 1).to(self.device)
            w[0, self.module_inp_indices[-1][0][0], 0] = 1.1
            w[0, self.module_inp_indices[-1][0][1], 0] = 1.0
        else:
            w = torch.randn(1, input_dim, 1).to(self.device)
        self.module_inp_embeddings.append(nn.Parameter(w))
        
        self.out = nn.Sigmoid()
    
    def forward(self, x, train=True):

        for lkj in range(len(self.num_img_slots)):

            module_input_embeddings = self.module_inp_embeddings[lkj]
            module_location_embeddings = self.module_loc_embeddings[lkj]

            if train:
                scores_inp = gumbel_softmax(module_input_embeddings, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
                scores = gumbel_softmax(module_location_embeddings, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
            else:
                scores_inp = gumbel_softmax(module_input_embeddings, hard = True, dim=1, use_gumbel=False, tau = None)
                scores = gumbel_softmax(module_location_embeddings, hard = True, dim=1, use_gumbel=False, tau = None)

            for i in range(self.num_img_slots[lkj]):

                s = scores_inp[:,:,i]
                s = s.repeat(28,1).repeat(28,1,1).permute(2, 0, 1).unsqueeze(0)
                slot_inp = (x*s).sum(dim=1).unsqueeze(1)#.view(x.size(0),1)
                slot_out = self.img_modules(slot_inp, scores[:,:,i])

                if i==0:
                    y = slot_out
                else:
                    y = torch.cat((y,slot_out), dim=1)
            x = y
        
        for lkj in range(len(self.num_slots)):
            
            module_input_embeddings = self.module_inp_embeddings[lkj+len(self.num_img_slots)]
            module_location_embeddings = self.module_loc_embeddings[lkj+len(self.num_img_slots)]

            if train:
                scores1, scores2 = gumbel_sigmoid(module_input_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
                scores = gumbel_softmax(module_location_embeddings, hard = True, dim=1, use_gumbel=self.use_gumbel, tau = self.tau)
            else:
                scores1, scores2 = gumbel_sigmoid(module_input_embeddings, k = self.topk, hard = True, dim=1, use_gumbel=False, tau = None)
                scores = gumbel_softmax(module_location_embeddings, hard = True, dim=1, use_gumbel=False, tau = None)

            for i in range(self.num_slots[lkj]):

                variable_1 = (x*scores1[:,:,i]).sum(dim=1).view(x.size(0),1)
                variable_2 = (x*scores2[:,:,i]).sum(dim=1).view(x.size(0),1)

                slot_inp = torch.cat((variable_1,variable_2), dim=1)
                slot_out = self.module_mlps(slot_inp)

                slot_out = (slot_out*scores[:,:,i]).sum(dim=1).view(x.size(0),1)

                if i==0:
                    y = slot_out
                else:
                    y = torch.cat((y,slot_out), dim=1)
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