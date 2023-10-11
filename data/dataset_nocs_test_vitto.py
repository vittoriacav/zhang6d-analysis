# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys
sys.path.insert(0,'third-party')

import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import json
import time
import pickle as pkl


class NOCSTestDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        self.repeat = opts.repeat


        with open(opts.test_list) as f:
            self.test_list = f.read().strip().split()
        self.imglist = []
        self.masklist = []
        self.depthlist = []
        self.metalist = []

        self.category_id_list = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}

        assert opts.test
        with open('/data/yangfu2/NOCS/obj_models/real_test.pkl', 'rb') as f_models:
            models = pkl.load(f_models)

        self.total_frames = 0

        for list_idx, seqname in enumerate(self.test_list):
            scene_index = eval(seqname)
            scene_list = sorted(os.listdir(opts.test_dataset_path))
            
            mask_list_total = glob.glob(os.path.join(opts.test_dataset_path, scene_list[scene_index], '*_mask.png'))
            mask_list_total.sort(key=lambda item: int(item.split('/')[-1].split('_')[0]))
            
            frame_obj_id_dict = {}
            for frame in range(len(mask_list_total)):
                with open(mask_list_total[frame].replace('_mask.png', '_meta.txt')) as f:
                    lines = f.read().strip().split('\n')
                    for ln in lines:
                        ln = ln.split()
                        if eval(ln[1]) == self.category_id_list[opts.category]:
                            if ln[2] not in frame_obj_id_dict.keys():
                                frame_obj_id_dict[ln[2]] = []
                            for item in frame_obj_id_dict:
                                if ln[2] == item:
                                    frame_obj_id_dict[ln[2]].append((frame, eval(ln[0])))
                                    break
            
            for obj_name in frame_obj_id_dict.keys():
                mask_list = []
                meta_list = []

                frame_obj_id_list = frame_obj_id_dict[obj_name]
                for frame, frame_obj_id in frame_obj_id_list:
                    mask_fn = mask_list_total[frame]
                    mask_list.append(mask_fn)
                    
                    meta_fn = mask_fn.replace('_mask.png', '_label.pkl')
                    with open(meta_fn, 'rb') as f:
                        data = pkl.load(f)
                        for iid in range(len(data['instance_ids'])):
                            if frame_obj_id == data['instance_ids'][iid]:

                                class_id = data['class_ids'][iid]
                                name = data['model_list'][iid]
                                assert class_id == self.category_id_list[opts.category]
                                assert name == obj_name
                                assert name in models.keys()

                                model = models[name]
                                scale = model.max(0) - model.min(0)
                                scale *= data['scales'][iid]

                                rotation = data['rotations'][iid]
                                rotation = rotation @ np.array([
                                        [1, 0, 0],
                                        [0, -1, 0],
                                        [0, 0, -1]
                                    ])
                                translation = data['translations'][iid]
                                bbox = data['bboxes'][iid]

                        meta = {'rotation': rotation, 
                                'translation': translation, 
                                'scale': scale,
                                'bbox': bbox,
                                'id': frame_obj_id}
                        meta_list.append(meta)

                img_list = [i.replace('_mask.png', '_color.png') for i in mask_list]
                depth_list = [i.replace('_mask.png', '_depth.png') for i in mask_list]

                self.imglist.append(img_list)
                self.masklist.append(mask_list)
                self.depthlist.append(depth_list)
                self.metalist.append(meta_list)

                self.total_frames += len(mask_list)

        self.transform = transforms.ToTensor()
        self.dframe = self.opts.dframe_eval
        total_samples = []
        for video_idx in range(len(self.masklist)):  # relative video id
            for i in range(0, len(self.masklist[video_idx]), self.dframe):
                total_samples.append((video_idx, i))  # n_videos * n_frames, 1

        self.sample_list = total_samples
        self.total_length = len(total_samples)
        print('total number of frames:', self.total_length)

    
    def __len__(self):
        return self.total_length
            

    def __getitem__(self, index):

        video_id, frame_id = self.sample_list[index]

        rand_scale = np.array([1.2, 1.2])

        ## start reading
        img = cv2.imread(self.imglist[video_id][frame_id])[:,:,::-1]
        mask = cv2.imread(self.masklist[video_id][frame_id], cv2.IMREAD_GRAYSCALE)
        if self.opts.use_depth:
            depth = cv2.imread(self.depthlist[video_id][frame_id], -1) * 1.0

        meta = self.metalist[video_id][frame_id]
        bbox = meta['bbox']
        
        frame_obj_id = meta['id']
        mask = (mask == frame_obj_id).astype(bool)

        scale = meta['scale']
        rotation = meta['rotation']
        translation = meta['translation']

        img = img * 1.0

        # crop box
        center = [int((bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2)]
        length = [int((bbox[3] - bbox[1]) / 2), int((bbox[2] - bbox[0]) / 2)]
        max_length = max(length[0], length[1])

        if self.opts.no_stretch: length = [int(rand_scale[0] * max_length), int(rand_scale[0] * max_length)]
        else: length = [int(rand_scale[0] * length[0]), int(rand_scale[1] * length[1])]
        foc = [int(591.0125), int(590.16775)]
        pp = [int(322.525), int(244.11084)]

        maxw = self.opts.img_size
        maxh = self.opts.img_size
        crop_factor  = [maxw / 2 / length[0], maxh / 2 / length[1]]
        foc_crop = [foc[0] * crop_factor[0], foc[1] * crop_factor[1]]
        pp_crop = [(pp[0]-(center[0]-length[0])) * crop_factor[0], (pp[1]-(center[1]-length[1])) * crop_factor[1]]

        img = self.transform(img) / 255.
        mask = torch.tensor(mask, dtype=torch.float32)[None]

        
        img = transforms.functional.resized_crop(img, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                    size=(maxh, maxw), interpolation=InterpolationMode.BILINEAR)
        mask = transforms.functional.resized_crop(mask, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                    size=(maxh, maxw), interpolation=InterpolationMode.NEAREST)


        if self.opts.use_depth:
            depth = torch.tensor(depth, dtype=torch.float32)[None]
            depth = transforms.functional.resized_crop(depth, center[1]-length[1], center[0]-length[0], 2*length[1], 2*length[0], \
                        size=(maxh, maxw), interpolation=InterpolationMode.NEAREST)

        elem = {
            'img': img,
            'mask': mask,
            'depth': depth if self.opts.use_depth else torch.zeros(1),
            'center': torch.tensor(center),
            'length': torch.tensor(length),
            'foc': torch.tensor(foc),
            'foc_crop': torch.tensor(foc_crop),
            'pp': torch.tensor(pp),
            'pp_crop': torch.tensor(pp_crop),
            'idx': torch.tensor([video_id]),
            'frame_idx': torch.tensor([frame_id]),
        }

        if self.opts.eval:
            elem['rotation'] = torch.tensor(rotation)
            elem['translation'] = torch.tensor(translation)
            elem['scale'] = torch.tensor(scale)

        return elem



        
        

