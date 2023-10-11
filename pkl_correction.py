import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd

categories = ["bowl", "camera", "bottle", "mug", "laptop"]

checkpoint_dir='log'
logname='exp25-kaifAUG_bowl_noUshape'
category = [cat for cat in categories if cat in logname][0]
if not category:
    category = 'bottle'
test = 'TestMixRot'
path="/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/{}/{}/{}/".format(checkpoint_dir, logname, test)

category = 'bowl'

with open(os.path.join(path, "_pca_data.pkl"), "rb") as file:
            data = pickle.load(file)

test_mixrot_degs_path = "/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/test_{}_mixRot_degs.txt".format(category)

for i in range(len(data['imgs_path'])):
    img_orig_path = data['imgs_path'][i]
    if os.path.exists(test_mixrot_degs_path):
                idx_start = img_orig_path.find(category)+len(category)+1
                idx_end = img_orig_path.find("images")
                target_key = img_orig_path[idx_start:idx_end]
                with open(test_mixrot_degs_path, 'r') as test_mixrot_degs:
                    for line in test_mixrot_degs:
                        # Check if the line starts with the target key
                        if line.startswith(target_key):
                            # Split the line using '/' as a delimiter
                            parts = line.strip().split('/')
                            # Extract the last part and remove the "deg_" prefix
                            result = int(parts[-1].replace("deg_", ""))
                            # Break out of the loop since we found the result
                            break
                data['degs'][i] = result
    else:
        print("path not exist")

with open(os.path.join(path, "_pca_data.pkl"), "wb") as file:
            pickle.dump(data, file)