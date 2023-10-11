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

def check_existence(path, test):
    if os.path.exists(path):
        print(path, " already exists")
        f = input("0 = overwrite, 1 = stop process:\t")
        
        if int(f) == 0:
            print("Removing previous...")
            shutil.rmtree(path)

            if test:
                #remove also associated pickles annotations
                shutil.rmtree(os.path.join(os.path.dirname(target), "pkl_annotations", category))
        else:
            print("Stopped Process")
            sys.exit()

def rotate_dataset(source, target, test, category, init_time):
    print(source)
    print(target)

    check_existence(target, test)
    print("Copying data...")
    shutil.copytree(source, target)

    degs = [0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90]

    list_instances = os.listdir(target)
    list_instances.sort()

    if test:
        log_degs_file = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/test_{}_mixRot_degs.txt'.format(category)    
        
        source_pkls = os.path.join(os.path.dirname(source), "pkl_annotations", category)
        target_pkls = os.path.join(os.path.dirname(target), "pkl_annotations", category)
        print("Copying annotations...")
        shutil.copytree(source_pkls, target_pkls)
    
    else:
        list_instances.remove("train_list.txt")
        log_degs_file = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/train_{}_mixRot_degs.txt'.format(category)    
        # log_degs_file = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/train_{}_90.txt'.format(category)    

    
    f = open(log_degs_file, 'w')
    f.close()

    # for deg in degs:
    #     f.writelines([instance,"\t", seq,"\t", str(deg)])
    #     f.write("\n")
    # f.close()

    print("Rotating...")
    for instance in list_instances:
        list_seqs = os.listdir(os.path.join(target, instance))
        list_seqs.sort()
        
        for seq in list_seqs:
            
            f = open(log_degs_file, 'a')
            f.writelines([instance,"/", seq,"/deg_", str(degs[0])])
            f.write("\n")
            f.close()

            imgs_seq_dir = os.path.join(target,instance,seq,"images")
            
            if degs[0] != 0:
                for file in os.listdir(imgs_seq_dir):
                    file_path = os.path.join(imgs_seq_dir,file)

                    if "depth" in file:
                        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    elif "mask" in file:
                        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    else:
                        img = cv2.imread(file_path)
                        
                    img_rotated = rotate(img, float(-degs[0]), preserve_range=True)           

                    if "depth" in file:
                        iio.imwrite(file_path, img_rotated.astype(np.uint16))
                    else:
                        cv2.imwrite(file_path, img_rotated)
                
                
                if test:
                    print("Modifying annotations...")
                    pkls_instance = '{}-{}-{}.pkl'.format(category, instance, seq)
                    pkls_path = os.path.join(target_pkls, pkls_instance)
                    
                    if os.path.isfile(pkls_path):
                        gt = pkl.load(open(pkls_path, 'rb'))
                        for obj in gt["annotations"]:

                            Rt_original = np.eye(4)
                            Rt_original[:3, :3] = obj["rotation"]
                            Rt_original[:3, 3] = obj["translation"]

                            Rt_deg = np.eye(4)
                            rdeg = R.from_euler('z', degs[0], degrees=True)

                            Rt_deg[:3, :3] = rdeg.as_matrix()


                            Rt_final =  Rt_deg @ Rt_original

                            obj["rotation"] = Rt_final[:3, :3]
                            obj["translation"] = Rt_final[:3, 3]

                        pkl.dump(gt, open(pkls_path, 'wb'))
                    else:
                        print("This annotation pkl file doesn't exist\n{}".format(pkls_path))

            degs.append(degs.pop(degs.index(degs[0])))

    end = datetime.now()
    print("total required time: ", end - init_time)


if __name__ == "__main__":

    test = True
    category = 'bottle'

    
    if test:
        dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/test_set/{}'.format(category)
        output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/test_mixRot/{}'.format(category)
    else:
        dataset_path = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}'.format(category)
        # output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}_mixRot'.format(category)
        output_path_rot = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/{}_mixRot'.format(category)
        

    init = datetime.now()
    
    rotate_dataset(dataset_path, output_path_rot, test, category, init)
    
    # see pickle
    # prova = pkl.load(open(os.path.join(output_path_rot, "pkl_annotations", category, "bottle-0001-1.pkl"), 'rb'))
    # print(prova["intrinsics"]) 
    