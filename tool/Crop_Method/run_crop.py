"""
@File    :   Face_Crop_Align.py
@Author  :   Kaiqing.Lin
@Update  :   2025/03/05
"""

import os
import os.path as osp
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.io as sio
import imageio.v2 as imageio
import cv2
from termcolor import cprint
from tqdm import tqdm
import time
from termcolor import cprint
from PIL import Image
from detect_face import extract_face


def extract(self, img, batch_boxes, save_path):
    # Determine if a batch or single image was passed
    batch_mode = True
    if (
            not isinstance(img, (list, tuple)) and
            not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
            not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
    ):
        img = [img]
        batch_boxes = [batch_boxes]
        batch_mode = False

    # Parse save path(s)
    if save_path is not None:
        if isinstance(save_path, str):
            save_path = [save_path]
    else:
        save_path = [None for _ in range(len(img))]

    # Process all bounding boxes
    faces = []
    for im, box_im, path_im in zip(img, batch_boxes, save_path):
        if box_im is None:
            faces.append(None)
            continue

        if not self.keep_all:
            box_im = box_im[[0]]

        faces_im = []
        for i, box in enumerate(box_im):
            face_path = path_im
            if path_im is not None and i > 0:
                save_name, ext = os.path.splitext(path_im)
                face_path = save_name + '_' + str(i + 1) + ext

            face = extract_face(im, box, self.image_size, self.margin, face_path)
            # if self.post_process:
            #     face = fixed_image_standardization(face)
            faces_im.append(face)

        if self.keep_all:
            faces_im = torch.stack(faces_im)
        else:
            faces_im = faces_im[0]

        faces.append(faces_im)

    if not batch_mode:
        faces = faces[0]

    return faces


def crop_forward(self, img, save_path=None, return_prob=False):
    """
    Run MTCNN face detection on a PIL image or numpy array. This method performs both
    detection and extraction of faces, returning tensors representing detected faces rather
    than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.

    Arguments:
        img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

    Keyword Arguments:
        save_path {str} -- An optional save path for the cropped image. Note that when
            self.post_process=True, although the returned tensor is post processed, the saved
            face image is not, so it is a true representation of the face in the input image.
            If `img` is a list of images, `save_path` should be a list of equal length.
            (default: {None})
        return_prob {bool} -- Whether or not to return the detection probability.
            (default: {False})

    Returns:
        Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
            with dimensions 3 x image_size x image_size. Optionally, the probability that a
            face was detected. If self.keep_all is True, n detected faces are returned in an
            n x 3 x image_size x image_size tensor with an optional list of detection
            probabilities. If `img` is a list of images, the item(s) returned have an extra
            dimension (batch) as the first dimension.

    Example:
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN()
    face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
    """

    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = extract(self, img, batch_boxes, save_path)

    if return_prob:
        return faces, batch_probs
    else:
        return faces


class PreProcess:
    def __init__(self, image_size=None, margin=80) -> None:
        super().__init__()
        from facenet_pytorch import MTCNN, InceptionResnetV1
        # Load MTCNN to Crop Face
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, device='cuda:5')

    def crop(self, img):
        img_cropped = self.mtcnn.forward(img, save_path=None)
        # img_cropped = crop_forward(self.mtcnn, img, save_path=None)
        return img_cropped

    def crop_square(self, img):
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
        batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    img, method="probability")
        scale = 1.2
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b) for b in bbox[:]]
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                face_size = max(xmax - xmin, ymax - ymin)
                size_bb = int(face_size * scale)

                left = max(center_x - size_bb // 2, 0)
                upper = max(center_y - size_bb // 2, 0)
                right = min(center_x + size_bb // 2, img.size[0])
                lower = min(center_y + size_bb // 2, img.size[1])

                img = np.array(img)
                crop = img[upper:lower, left:right]
                img = Image.fromarray(crop)
                return img

    def __call__(self, img_dir):
        # 1. 找为什么会产生缩放的情况

        image = Image.open(img_dir)

        # image = image.resize((1024, 1024))
        # img_aligned = self.align(img)   # 对齐
        # img_cropped = self.crop(image)    # 裁剪
        img_cropped = self.crop_square(image)    # 裁剪
        return img_cropped


if __name__ == '__main__':
    data_dir = '/data4/linkaiqing/code/VIP_Benchmark_Img_V2/Gen_Img/'
    
    crop_model = PreProcess()
    base_save_dir = '/data4/linkaiqing/code/VIP_Benchmark_Img_Final_Version/Gen_Img/'
    
    for method in ['efs_gpt4o']:
        for id in range(0, 22):
            file_path = os.path.join(data_dir, method, f'id{str(id)}', 'Real_Test')
            name_list = os.listdir(file_path)
            save_name_path = os.path.join(base_save_dir, method, f'id{str(id)}')
            if not os.path.exists(save_name_path):
                os.makedirs(save_name_path)
            # 移除.DS_Store文件
            if '.DS_Store' in name_list:
                name_list.remove('.DS_Store')
            
            for img_path in tqdm(name_list, desc=f'{method} id{str(id)}'):
                img_dir = os.path.join(file_path, img_path)
                x = crop_model(img_dir)
                img = x
                img.save(os.path.join(save_name_path, img_path.replace('.jpeg', '.png').replace('.jpg', '.png')))

            
                
    # for name in name_list:
    #     name_path = os.path.join(data_dir, name)
    #     save_name_path = os.path.join(base_save_dir, name)
    #     if not os.path.exists(save_name_path):
    #         os.makedirs(save_name_path)

    #     img_list = os.listdir(name_path)

    #     for img_path in tqdm(img_list):
    #         try:
    #             img_dir = os.path.join(name_path, img_path)
    #             x = crop_model(img_dir)
    #             img = x
    #             img.save(os.path.join(save_name_path, img_path.replace('.jpeg', '.png').replace('.jpg', '.png')))
    #         except:
    #             print(img_dir)
    #             continue