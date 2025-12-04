import torch
import numpy as np
import torch.nn as nn


class FG_Face(nn.Module):
    def __init__(self, attributes, token_dim, model_name='transface'):
        super(FG_Face, self).__init__()
        self.attr = attributes
        self.learnable_attr_params = []

        for a in self.attr:
            # 按属性添加fc
            self.__setattr__(f'{a}_fc', nn.Linear(512, token_dim, bias=True))

            # 添加可学习参数
            self.learnable_attr_params.append(getattr(self, f"{a}_fc", None).weight)
            if getattr(self, f"{a}_fc", None).bias is not None:
                self.learnable_attr_params.append(getattr(self, f"{a}_fc", None).bias)

        # 对齐维度
        self.token_dim = token_dim
        # 加载人脸识别模型
        self.model_name = model_name
        self.face_model = self.load_model(self.model_name)
        self.face_model.eval()

    def load_model(self, model_name='transface'):
        if model_name == 'transface':
            from Models.Face_Model.TransFace.backbones import get_model
            model = get_model('vit_l_dp005_mask_005')

            pretrained = './Models/Face_Model/checkpoints/transface/glint360k_model_TransFace_L.pt'
            model_dict = torch.load(pretrained, map_location='cpu')
            model.load_state_dict(model_dict)
            model.eval()
            print("Load Face Model Successfully, The Model is TransFace")
        if model_name == 'arcface':
            from Models.Face_Model.ArcFace.Load_Model import load_arcface

            model = load_arcface()

        return model

    @torch.no_grad()
    def face_emb_forward(self, x):
        self.face_model.eval()
        if self.model_name == 'transface':
            feat, _, _ = self.face_model(x)
        else:
            feat = self.face_model(x)
        return feat

    def forward(self, x, face_emb=None):
        self.face_model.eval()
        face_tokens = torch.zeros((len(self.attr), self.token_dim)).cuda()

        # 若face_emb为None，则进行人脸识别
        if face_emb is None:
            with torch.no_grad():
                if self.model_name == 'transface':
                    feat, _, _ = self.face_model(x)
                else:
                    feat = self.face_model(x)

        # 人脸识别后，提取人脸细粒度特征
        for i, a in enumerate(self.attr):
            face_tokens[i] = getattr(self, f"{a}_fc", None)(face_emb)

        return face_tokens

    def get_learnable_params(self):
        return self.learnable_attr_params

    def load_face_model(self, net_state_dict):
        attr = net_state_dict['attr']
        for a in attr:
            getattr(self, f"{a}_fc", None).load_state_dict(net_state_dict[a])
        print("The attribute of the face model is: ", attr)
        print("Load Face Model Successfully")

    def save_face_model(self):
        save_dict = {}
        for attr in self.attr:
            save_dict[attr] = getattr(self, f"{attr}_fc", None).state_dict()
        save_dict['attr'] = self.attr
        return save_dict
