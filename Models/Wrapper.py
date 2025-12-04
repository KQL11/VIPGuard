'''
@File    :   Warpper.py
@Author  :   Kaiqing.Lin 
@Update  :   2024/12/17
'''
import torch
import torch.nn as nn
from typing import Union


class Wrapper(nn.Module):
    def __init__(self, vl_model, processor, optim_facechecker=False, vip_token_num=196):
        super(Wrapper, self).__init__()

        self.vl_model = vl_model
        self.vl_model.requires_grad_(False)
        self.processor = processor
        self.tokenizer = processor.tokenizer

        vip_token_dim = self.vl_model.facechecker.face_checker.vip_prompt.shape[1]
        self.optim_facechecker = optim_facechecker
        param = torch.empty([vip_token_num, vip_token_dim], device='cuda')
        param = nn.Parameter(nn.init.normal(param, mean=0, std=1))
        self.vl_model.facechecker.face_checker.vip_prompt = param

    def get_optim_params(self):
        optim_params = []
        for n, p in self.vl_model.named_parameters():
            if self.optim_facechecker:
                if 'facechecker' in n:
                    p.requires_grad_(True)
                    optim_params.append(p)
                    print(n)
            else:
                if 'vip_prompt' in n:
                    optim_params.append(p)
                    print(n)

        return optim_params

    def save_vip(self):
        if self.optim_facechecker:
            return {
                'vip_token': self.vl_model.facechecker.face_checker.vip_prompt,
                'facechecker': self.vl_model.facechecker.state_dict(),
            }
        else:
            return {
                'vip_token': self.vl_model.facechecker.face_checker.vip_prompt
            }

    def load_vip(self, pretrain:Union[str, dict]):
        if isinstance(pretrain, str):
            state_dict = torch.load(pretrain, map_location='cpu')
            if self.optim_facechecker:
                self.vl_model.facechecker.load_state_dict(state_dict['facechecker'])
            else:
                self.vl_model.facechecker.face_checker.vip_prompt.data.copy_(state_dict['vip_token'])
        elif isinstance(pretrain, dict):
            if self.optim_facechecker:
                self.vl_model.facechecker.load_state_dict(pretrain['facechecker'])
            else:
                self.vl_model.facechecker.load_state_dict(pretrain['vip_token'])
        print("Loading Pre-trained VIP Token!")

    def forward(self, **kwargs):
        return self.vl_model.forward(**kwargs)

    def generate(self, **kwargs):
        return self.vl_model.generate(**kwargs)
