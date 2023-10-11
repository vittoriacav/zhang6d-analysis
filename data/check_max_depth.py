import os
import numpy as np
import cv2

target = "/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/wild6d/bottle_mixRot"

list_instances = os.listdir(target)
list_instances.sort()
list_instances.remove("train_list.txt")

best_max = 0

for instance in list_instances:
        list_seqs = os.listdir(os.path.join(target, instance))
        list_seqs.sort()
        
        for seq in list_seqs:
            imgs_seq_dir = os.path.join(target,instance,seq,"images")
            
            for file in os.listdir(imgs_seq_dir):
                    file_path = os.path.join(imgs_seq_dir,file)

                    if "depth" in file:
                        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

                        current_max = img.max()

                        if current_max > best_max:
                            best_max = current_max



print(best_max)