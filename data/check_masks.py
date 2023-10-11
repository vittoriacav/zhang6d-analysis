import os
import glob
import numpy as np
import cv2
import random 
import sys 
import pickle as pkl
import json
import shutil
import matplotlib.pyplot as plt
from skimage.transform import rotate
import imageio as iio
from scipy.spatial.transform import Rotation as R
from datetime import datetime

init = datetime.now()

multi_cat = False

if multi_cat:
    tests = [True, False]
    categories = ['bottle', 'bowl', 'camera', 'laptop', 'mug']
    stats = {'bottle':{'train':0, 'test': 0, 'avgs': []}, 'bowl':{'train':0, 'test': 0, 'avgs': []}, 'camera':{'train':0, 'test': 0, 'avgs': []},
            'laptop':{'train':0, 'test': 0, 'avgs': []}, 'mug':{'train':0, 'test': 0, 'avgs': []}}


    for category in categories:
        cat_sum1 = 0
        cat_sum2 = 0

        for test in tests:

            if test:
                dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set/{}'.format(category)
                split = 'test'
            else:
                dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}'.format(category)
                split = 'train'

            file_list = glob.glob(dataset_path+'/**/*-mask.png', recursive = True)

            os.makedirs("/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}/{}".format(category,split), exist_ok = True)
            writing_dir = "/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}/{}".format(category,split)

            c = 0
            tot_sum1 = 0
            tot_sum2 = 0
            len_list2 = 0
            # dist_data = []
            for file_path in file_list:
                head_tail_mask = os.path.split(file_path)
                mask_name = head_tail_mask[1]
                img_name = mask_name.split("-")[0]+".jpg"
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                size_px = np.sum(mask==255)
                # dist_data.append(size_px)
                tot_sum1 += size_px
                if size_px < 2000:
                    writing_path_mask = os.path.join(writing_dir, str(c)+"-mask.png")
                    img_orig_path = os.path.join(head_tail_mask[0],img_name)
                    writing_path_img = os.path.join(writing_dir, str(c)+".jpg")
                    cv2.imwrite(writing_path_mask, mask)
                    shutil.copy(img_orig_path, writing_path_img)
                    print(c, " : found one! size = ", size_px)
                    c += 1
                else:
                    tot_sum2 += size_px
                    len_list2 += 1

            stats[category][split] = c

            cat_sum1 += tot_sum1/len(file_list)
            cat_sum2 += tot_sum2/len_list2

            # dist_data = np.array(dist_data)
            # np.save('/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}_{}.npy'.format(category, split), dist_data)
            # fig = plt.figure()
            # max_dist = dist_data.max()
            # plt.hist(dist_data, bins = np.arange(max_dist, step = max_dist//50) )
            # plt.title("Mask Pixel Size Distribution in {} category ({} set)".format(category, split))
            # plt.xlabel('Mask Pixel Size')
            # plt.ylabel('Density')
            # plt.savefig("/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}_{}.png".format(category, split))
        
        stats[category]['avgs'] = [cat_sum1/2, cat_sum2/2]
        print(stats)
else:
    test = False
    category = 'mug' 

    if test:
        dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set/{}'.format(category)
        split = 'test'
    else:
        dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}'.format(category)
        split = 'train'

    file_list = glob.glob(dataset_path+'/**/*-mask.png', recursive = True)

    os.makedirs("/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}/{}".format(category,split), exist_ok = True)
    writing_dir = "/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/wrong/{}/{}".format(category,split)

    c = 0
    for file_path in file_list:
        head_tail_mask = os.path.split(file_path)
        mask_name = head_tail_mask[1]
        img_name = mask_name.split("-")[0]+".jpg"
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        size_px = np.sum(mask==255)
        if size_px < 9365 or size_px > 63682:
            writing_path_mask = os.path.join(writing_dir, str(c)+"-mask.png")
            img_orig_path = os.path.join(head_tail_mask[0],img_name)
            writing_path_img = os.path.join(writing_dir, str(c)+".jpg")
            cv2.imwrite(writing_path_mask, mask)
            shutil.copy(img_orig_path, writing_path_img)
            if size_px < 9365 and size_px > 8800:
                print(c, " : found one! size = ", size_px)
            if size_px > 63682:
                print(c, " : found one! size = ", size_px)
            c += 1



end = datetime.now()
print("total required time: ", end - init)
            