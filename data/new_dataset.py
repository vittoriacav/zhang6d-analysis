import os
import numpy as np
import cv2
import random 
import sys 
import pickle as pkl
import json
import shutil
from skimage.transform import rotate
import imageio as iio
from scipy.spatial.transform import Rotation as R

def check_existence(path):
    if os.path.exists(path):
        print(path, " already exists")
        f = input("0 = overwrite, 1 = stop process:\t")
        
        if int(f) == 0:
            print("Overwriting...")
            shutil.rmtree(path)
        else:
            print("Stopped Process")
            sys.exit()

def rotate_dataset(source, target, category, deg, test=True, change_size=False):
    
    check_existence(target)

    shutil.copytree(source, target)

    print("Rotating...")
    for (root,dirs,files) in os.walk(os.path.join(target, category), topdown=True):
    
        if files and files[0] != "metadata" and files[0] != "icon":
            
            for file in files:
                img_path = os.path.join(root,file)
                print(img_path)
                if "depth" in file:
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                elif "mask" in file:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(img_path)
                    
                img_rotated = rotate(img, float(-deg), preserve_range=True)           

                if "depth" in file:
                    iio.imwrite(img_path, img_rotated.astype(np.uint16))
                else:
                    cv2.imwrite(img_path, img_rotated)

        
        # to modify metadata
        # elif files and files[0] == "metadata": 
        #     meta_path = os.path.join(root,files[0])
        #     meta = json.load(open(meta_path, 'r+'))
        #    
        #     with open(meta_path, 'w+') as outfile:
        #         json_string = json.dumps(meta)
        #         outfile.write(json_string)
 
    if test:
        print("Modifying annotations...")
        pkls_path = os.path.join(target, "pkl_annotations", category)

        for ann in os.listdir(pkls_path):
            ann_path = os.path.join(pkls_path, ann)
            gt = pkl.load(open(ann_path, 'rb'))
            for obj in gt["annotations"]:

                Rt_original = np.eye(4)
                Rt_original[:3, :3] = obj["rotation"]
                Rt_original[:3, 3] = obj["translation"]

                Rt_deg = np.eye(4)
                rdeg = R.from_euler('z', deg, degrees=True)

                Rt_deg[:3, :3] = rdeg.as_matrix()


                Rt_final =  Rt_deg @ Rt_original

                obj["rotation"] = Rt_final[:3, :3]
                obj["translation"] = Rt_final[:3, 3]

            pkl.dump(gt, open(ann_path, 'wb'))

         

def generate_subset(ds_path, out_path, category, n_objs, rand = False, test = True):
    """
    ds_path = source dataset path
    out_path = output subset path
    n_objs = number of instances to sample 
    rand = sample instances randomly?
    """
    check_existence(out_path)

    os.makedirs(out_path)
    if test:
        inst_dir = os.path.join(ds_path, category)
    else:
        inst_dir = ds_path
    list_instances = os.listdir(inst_dir)
    list_instances.sort()
    
    if rand: 
        instances = random.sample(list_instances[:-1], n_objs)
    else:
        instances = list_instances[:n_objs]


    for instance in instances: 
        # instance = list_instances[index]
        if test:
            source = os.path.join(ds_path, category, instance)
            target = os.path.join(out_path, category, instance)
        else:
            source = os.path.join(ds_path, instance)
            target = os.path.join(out_path, instance)
        
        shutil.copytree(source, target)

        if test:
            # copy pkl annotations too
            ann_dir = os.path.join(ds_path, 'pkl_annotations', category)
            all_pkls = os.listdir(ann_dir)

            pkls_instance = [i for i in all_pkls if instance in i]
    
            
            pkl_out_path = os.path.join(out_path,'pkl_annotations', category)
            
            os.makedirs(pkl_out_path, exist_ok = True)
            
            for pkl in pkls_instance:
                source = os.path.join(ann_dir, pkl)
                target = os.path.join(pkl_out_path, pkl)
                shutil.copy(source, target)

    instances.sort()
    print("Generated dataset with {} objects: {}".format(n_objs, instances))
        
            

if __name__ == "__main__":

    # number of object instance to create subset:
    # ex. n_objs = 5 --> 5 folders (one per instance) containing three video clip each
    n_objs = 3
    rand = False  # sample n_objs randomly or not?
    test = True
    deg = 90
    category = 'bottle'

    dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/Wild6D/bottle'
    output_path_sub = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/Wild6D/test_set'
    output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/Wild6D/test_set{}'.format(deg)

    # generate_subset(dataset_path, output_path_sub, category, n_objs, rand, test)
    rotate_dataset(output_path_sub, output_path_rot, category, deg, test, change_size=False)
    
    # see pickle
    # prova = pkl.load(open(os.path.join(output_path_rot, "pkl_annotations", category, "bottle-0001-1.pkl"), 'rb'))
    # print(prova["intrinsics"]) 
    