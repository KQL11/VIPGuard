'''
@File    :   VIP_Dataset.py
@Author  :   Kaiqing.Lin
@Update  :   2025/05/01
'''
import os
import torch
from torch.utils.data import DataLoader, Dataset
import copy


class VIP_Dataset(Dataset):
    def __init__(self, json_path, processor):
        """
        This dataset is used to load the facial image pairs.
        
        Args:
            json_path: The path of json (training information)
            processor: The pre-processor of the target multi-modal large language models (Qwen 2.5 7B)
        """
        self.json_path = json_path
        self.json_file = self.read_json(json_path)  # Get data from json file
        self.processor = processor

    def __len__(self):
        return self.json_file.__len__()

    def _gen_question_from_template(self, template):
        """
        This function is used to build a question according to the default template
        """
        template = copy.deepcopy(template)
        question = template[0]
        answer = template[1]
        answer_empty = {
            "role": "assistant",
            "content": [
                {"type": "text", "text":
                    ""
                    },
            ],
        }

        return question, answer, answer_empty

    def load_data(self, json_data):
        """
        This function is used to build data according to the default template
        """
        prompt = json_data['messages'][0]['content']
        prompt = prompt.replace('<image>', '')
        img_dir = json_data['images'][0]    # The first image is used
        response = json_data['messages'][1]['content']
        chat = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image',
                        'image': img_dir
                    },
                    {
                        'type': 'text',
                        'text': prompt
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': response
                    }
                ]
            }
        ]
        if 'r_r_i' in img_dir.lower():    # load the label according to the type of pairs
            label = 0
        else:
            label = 1
            
        question = chat[0]
        answer = chat[1]
        answer_empty = {
            "role": "assistant",
            "content": [
                {"type": "text", "text":
                    ""
                 },
            ],
        }

        return question, answer, answer_empty, img_dir, label

    def load_train_data(self, idx):
        """
        Load training data
        
        """
        json_data = self.json_file[idx]
        question, answer, answer_empty, img_dir, label = self.load_data(json_data)
        face_emb_dir = None
        img = torch.zeros((3, 224, 224))

        if face_emb_dir is not None:
            # Load face embedding 
            face_emb = torch.load(face_emb_dir)['face_emb']
        else:
            # If face embedding is None, return None and extract face embedding
            face_emb = None

        label = torch.tensor(label).long()

        # Generate message and question
        message = [question, answer]
        question = [question, answer_empty]

        return message, question, label, img_dir, img, face_emb

    def read_json(self, json_path):
        import json
        with open(json_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)  # Parse JSON data
        return data

    def __getitem__(self, idx):
        message_ori, question_ori, label, img_dir, img, face_emb = self.load_train_data(idx)
        message = self.processor.apply_chat_template(message_ori, tokenize=False, add_generation_prompt=False, our_token=True, our_token_length=1)
        question = self.processor.apply_chat_template(question_ori, tokenize=False, add_generation_prompt=True, our_token=True, our_token_length=1)

        data = {
            'message': message,
            'question': question,
            'message_ori': message_ori,
            'question_ori': question_ori,
            'label': label,
            'img': img,
            'img_dir': img_dir,
        }

        if face_emb is not None:
            data['face_emb'] = face_emb

        return data

