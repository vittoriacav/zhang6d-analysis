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
from datetime import datetime
from MiDaS.run import run

def check_existence(path, test):
    if os.path.exists(path):
        print(path, " already exists")
        f = input("0 = overwrite, 1 = stop process:\t")
        
        if int(f) == 0:
            print("Removing previous...")
            shutil.rmtree(path)

            if test:
                #remove also associated pickles annotations
                shutil.rmtree(os.path.join(os.path.dirname(path), "pkl_annotations", category))

        else:
            print("Stopped Process")
            sys.exit()

def predict_depth(source, target, test, category, init_time):
    print(source)
    print(target)

    check_existence(target,test)
    print("Copying data...")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*depth*"))

    list_instances = os.listdir(target)
    list_instances.sort()


    if test:        
        source_pkls = os.path.join(os.path.dirname(source), "pkl_annotations", category)
        target_pkls = os.path.join(os.path.dirname(target), "pkl_annotations", category)
        print("Copying annotations...")
        shutil.copytree(source_pkls, target_pkls)
    else:
        list_instances.remove("train_list.txt")

    print("Estimating depth...")
    for instance in list_instances:
        list_seqs = os.listdir(os.path.join(target, instance))
        list_seqs.sort()
        
        for seq in list_seqs:
         
            imgs_seq_dir = os.path.join(target,instance,seq,"images")
            
            # run(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height, args.square, args.grayscale)
            run(imgs_seq_dir, imgs_seq_dir, "/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/MiDaS/weights/dpt_beit_large_512.pt", \
                    "dpt_beit_large_512", False, False, None, False, True)
                  
   
    end = datetime.now()
    print("total required time: ", end - init_time)


if __name__ == "__main__":

    test = True
    category = 'bottle'

    
    if test:
        dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set/{}'.format(category)
        output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set_DEPTH/{}'.format(category)
    else:
        dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}'.format(category)
        # output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}_mixRot'.format(category)
        output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}_DEPTH'.format(category)
        
    init = datetime.now()
    predict_depth(dataset_path, output_path_rot, test, category, init)
    
    # see pickle
    # prova = pkl.load(open(os.path.join(output_path_rot, "pkl_annotations", category, "bottle-0001-1.pkl"), 'rb'))
    # print(prova["intrinsics"]) 
    