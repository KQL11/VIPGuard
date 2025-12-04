'''
@File    :   I-Face_Data_preprocess.py
@Author  :   Kaiqing.Lin 
@Update  :   2025/03/09
'''

import os
import os.path as osp
import numpy as np
import cv2
from skimage import transform as trans
from tqdm import tqdm
import dlib


class face_align():
    def __init__(self, ):
        super(face_align, self).__init__()
        self.lmk_detector, self.lmk_predictor = self._init_dlib_lm()
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src

    def _init_dlib_lm(self, path=None):
        # 初始化dlib的人脸检测器和关键点检测器
        if path is None:
            predictor_path = '/pubdata/linkaiqing/code/VIP_Benchmark_Img_V2/Reference_Img/utils/shape_predictor_68_face_landmarks.dat'
        else:
            predictor_path = path
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        return detector, predictor

    def resize_src(self, new_x, new_y):
        # 计算缩放比例
        scale_x = new_x / 112.0
        scale_y = new_y / 112.0

        # 计算新的 `src`
        new_src = self.src * np.array([scale_x, scale_y], dtype=np.float32)
        return new_src

    def get_lmk(self, img):
        dets = self.lmk_detector(img, 1)
        # assert len(dets) == 1, "The number of face should be 1."
        for k, d in enumerate(dets):
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            #     k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            # shape = predictor(img, d)
            lmk = self.lmk_predictor(img, d)
            lmk = ([[p.x, p.y] for p in lmk.parts()])
            break
        return lmk
    
    def align_face(self, rimg, return_lmk=False):
        # 对齐人脸
        # 1. 旋转
        # 2. 缩放
        # 3. 平移
        # 4. 裁剪
        h, w = rimg.shape[:2]
        if h < 200 or w < 200:
            return None
        else:
            new_w, new_h = w, h

        for iter_ in range(10):
            try:
                landmark = self.get_lmk(rimg)
                landmark = np.array(landmark, dtype=np.float32)

                assert landmark.shape[0] == 68 or landmark.shape[0] == 5
                assert landmark.shape[1] == 2
                if landmark.shape[0] == 68:
                    landmark5 = np.zeros((5, 2), dtype=np.float32)
                    landmark5[0] = (landmark[36] + landmark[39]) / 2
                    landmark5[1] = (landmark[42] + landmark[45]) / 2
                    landmark5[2] = landmark[30]
                    landmark5[3] = landmark[48]
                    landmark5[4] = landmark[54]
                else:
                    landmark5 = landmark
                    
                # 重新计算 `src`
                new_src = self.resize_src(new_w, new_h)
                tform = trans.SimilarityTransform()
                tform.estimate(landmark5, new_src)
                M = tform.params[0:2, :]
                    
                img = cv2.warpAffine(rimg,
                                    M, (new_w, new_h),
                                    borderValue=0.0)
                if return_lmk:
                    return img, landmark5
                else:
                    return img
            except:
                if return_lmk:
                    return None, None
                else:
                    return None

    def align_face_by_lmk(self, rimg, lmk5):
        # 对齐人脸
        # 1. 旋转
        # 2. 缩放
        # 3. 平移
        # 4. 裁剪
        if lmk5 is None:
            return None
        h, w = rimg.shape[:2]
        if h < 200 or w < 200:
            return None
        else:
            new_w, new_h = w, h


        landmark = lmk5
        
        # 重新计算 `src`
        new_src = self.resize_src(new_w, new_h)
        tform = trans.SimilarityTransform()
        tform.estimate(landmark, new_src)
        M = tform.params[0:2, :]
            
        img = cv2.warpAffine(rimg,
                            M, (new_w, new_h),
                            borderValue=0.0)

        return img


if __name__ == '__main__':
    # 重新处理脸部数据
    # 1. 对齐 
    # 2. 长宽尺寸小于200的图像，直接去掉
    
    # 只要正脸?
    face_aligner = face_align()
    
    base_base_path = '/pubdata/linkaiqing/code/VIP_Benchmark_Img_V2/Gen_Img/'
    base_save_base_path = '/pubdata/linkaiqing/code/VIP_Benchmark_Img_V2/Gen_Img_Align'

    # method_list = os.listdir(base_base_path)
    method_list = ['efs_gpt4o', 'efs_kling', 'efs_tongyi', 'efs_jimeng']
    for method in method_list:
    # for method in ['fs_mobilefaceswap']:
        base_path = osp.join(base_base_path, method)
        save_base_path = osp.join(base_save_base_path, method)
        name_list = os.listdir(base_path)

        for type_name in name_list:
            img_list = os.listdir(osp.join(base_path, type_name))
            for img_name in tqdm(img_list, desc='Processing %s' % type_name, ncols=100):
                lmk_5 = None
                save_dict = {}
                ori_dict = {}

                img_path = osp.join(base_path, type_name, img_name)
                save_img_path = osp.join(save_base_path, type_name, img_name)
                if os.path.exists(save_img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    print('Error:', img_path)
                    continue

                align_img = face_aligner.align_face(img)
                save_dict[save_img_path] = align_img
                ori_dict[save_img_path] = img
                    
                save_flag = True
                for key in save_dict.keys():
                    align_img = save_dict[key]
                        
                    if not osp.exists(osp.dirname(save_img_path)):
                        os.makedirs(osp.dirname(save_img_path))
                    # for key in save_dict.keys():
                    #     cv2.imwrite(key, save_dict[key])
                                        
                    if align_img is None:
                        img = ori_dict[key]
                    else:
                        img = save_dict[key]

                    cv2.imwrite(key, img)


