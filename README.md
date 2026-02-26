# VIP DFD LLM - Visual Identity Protection for Deepfake Detection

<p align="center">
  <a href="https://arxiv.org/abs/2505.19582"><img src="https://img.shields.io/badge/ArXiv-B31B1B?logo=arxiv&logoColor=white" alt="ArXiv"></a>
  <a href="https://github.com/KQL11/VIPGuard"><img src="https://img.shields.io/badge/Code-7F52FF?logo=github&logoColor=white" alt="Code"></a>
  <a href="https://huggingface.co/datasets/Kaiqing/VIPBench/tree/main"><img src="https://img.shields.io/badge/Dataset-0d69c5?logo=huggingface" alt="Dataset"></a>
  <a href="https://huggingface.co/Kaiqing/VIP-Guard"><img src="https://img.shields.io/badge/Model-369c2b?logo=huggingface" alt="Model"></a>
</p>

This is the official implementation of VIP-Guard for personalized deepfake detection using multimodal large language models (MLLM). The Stage 3 training script focuses on fine-tuning the model for specific face identity protection.


## 🌟 Highlights

- **Multimodal Deepfake Detection**: Uses MLLM approach for advanced deepfake detection
- **Identity-Specific Protection**: Focuses on protecting specific face identities from deepfakes

## 📕 TODO List
 - [x] Release the Stage 3 training code.
 - [x] Release [VIP-Eval dataset](https://huggingface.co/datasets/Kaiqing/VIPBench/tree/main).
 - [x] Release the checkpoint of [VIP-Guard](https://huggingface.co/Kaiqing/VIP-Guard) pre-trained on Stage 1 and 2.
 - [x] Release the **Stage 1,2** training code.
 - [x] Release the checkpoints of **VIP Tokens** for 22 IDs.
 - [x] Release the **training dataset** on the Stage 3.
 - [x] Release the **inference code** of VIP-Guard.

## 🚀 Quick Start Guide

### Installation

```bash
git clone https://github.com/KQL11/VIPGuard.git
cd VIPGuard
pip install -r requirements.txt
cd ms-swift
pip install -e .
cd requirements
bash install_all.sh
pip install qwen_vl_utils==0.0.11
```

### Download Face Model
We select Transface as the face model in our paper. 
Please download the checkpoint (glint360k_model_TransFace_L.pt) from [this link](https://drive.google.com/file/d/1jXL_tidh9KqAS6MgeinIk2UNWmEaxfb0/view?pli=1).
The checkpoint should be placed at 
```
./Face_Model/checkpoints/transface/glint360k_model_TransFace_L.pt
```

### Download VIP-Guard Model (pretrained on Stage 1 and 2)
Please download the pre-trained checkpoint of VIP-Guard from [this link](https://huggingface.co/Kaiqing/VIP-Guard).
The checkpoint should be placed at 
```
./checkpoints/checkpoints_attr_Stage2_merge
```

The pre-trained vip tokens are placed in `./FaceDATA/Pretrained_VIPToken`

### Download Training DATA (Stage 3)
We prepare the training data in Stage 3 at [this link](https://huggingface.co/datasets/Kaiqing/VIPGuard_Stage3_DATA)
Please download the data and place them in `./FaceDATA/Training_Img`.

---

### Customization for Your Data
##### Data Preprocessing
Face images need to be cropped and aligned before training:

- **Face Cropping**: Use the methods in `./tool/Crop_Method/run_crop.py`
- **Face Alignment**: Use the methods in `./tool/Align_Method/face_align.py`
- **Face Embedding Center**: For each ID, create a Face Embedding Center from the training set (compute center of all face vectors), then use this center to calculate similarity scores as question input. (For our data, we have placed the face center embeddings in `./FaceDATA/FaceEmb_Center`)

##### Training Dataset Preparation
Create JSON files for training with the following structures:

- **Images only**: Follow `./Example/id0_only_img.json`
- **Images + Text**: Follow `./Example/id0_w_text.json`

---

### Usage
We provide an example of Stage 3 training.  

```bash
cd VIPGuard

python Stage3_train_en.py \
    --name Amair \
    --train_json_path ./Example/id0_only_img.json \
    --device 0 \
    --epoch 1 \
    --token_num 32 \
    --gradient_accumulation_step 8
```

### Key Parameters

- `--name`: Training ID for model identification (default: 'Amair')
- `--device`: CUDA device ID to use (default: '3')
- `--train_json_path`: Path to training dataset JSON (default: './Example/id0_only_img.json')
- `--epoch`: Number of training epochs (default: 1)
- `--token_num`: Number of VIP tokens to use (default: 32)
- `--gradient_accumulation_step`: Steps to accumulate gradients (default: 8)

### Training Configuration

Default training settings are defined in the configuration constants:

```python
DEFAULT_CONFIG = {
    'MAX_IMAGE_SIZE': 448,        # Maximum image size
    'DEFAULT_LR': 1.0,            # Default learning rate
    'WEIGHT_DECAY': 1e-3,         # Weight decay
    'MAX_NEW_TOKENS': 4096,       # Maximum newly generated tokens
    'PRETRAIN_PATH': './checkpoints/checkpoints_attr_Stage2_merge'
}
```

### Model Checkpoints

The training process automatically manages model checkpoints:

- **Checkpoint Path**: `./checkpoints/Stage3/{name}/`
- **Model State**: `vip_token.pt` - Contains the VIP token states
- **Automatic Skip**: Training is skipped if checkpoint already exists


## 🔍 Inference & Testing Guide

Run inference (```Inference.ipynb```) on test images using the trained model.


## 📄 Citation

If you find this project helpful for your research, please cite our work:

```bibtex
@article{lin2025guard,
  title={Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes},
  author={Lin, Kaiqing and Yan, Zhiyuan and Zhang, Ke-Yue and Hao, Li and Zhou, Yue and Lin, Yuzhen and Li, Weixiang and Yao, Taiping and Ding, Shouhong and Li, Bin},
  journal={arXiv preprint arXiv:2505.19582},
  year={2025}
}
```

## 📬 Contact & Feedback

For questions or feedback, please reach out:

- **Email**: linkaiqing2021@email.szu.edu.cn

---

⭐️ If this repository helped your research, please star 🌟 this repo 👍!

## Acknowledgement
We gratefully acknowledge the following repositories, which our implementation builds upon:

- [Transface](https://github.com/DanJun6737/TransFace)
- [Ms-Swift](https://github.com/modelscope/ms-swift/tree/main)
- [Transformers](https://github.com/huggingface/transformers)

