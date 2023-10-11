import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio as iio

source = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/FORTHESIS/midas_depth/bottle'
target = '/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/FORTHESIS/midas_depthINV/bottle'

list_instances = os.listdir(source)
list_instances.sort()
print(list_instances)

for instance in list_instances:
    file_path = os.path.join(source,instance)
    dest = os.path.join(target,instance)
    
    if "depth" in instance:
        depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        print(depth.dtype)
        depth = 4000-depth
        iio.imwrite(dest, depth.astype(np.uint8))
                        


# gt = cv2.imread("./data/wild6d/test_set/bottle/0001/1/images/0-depth.png", cv2.IMREAD_UNCHANGED)
# pred = cv2.imread("./data/wild6d/test_mixRot_DEPTH/bottle/0001/1/images/0-depth.png", cv2.IMREAD_UNCHANGED)
# mask = cv2.imread("./data/wild6d/test_mixRot/bottle/0001/1/images/0-mask.png", cv2.IMREAD_UNCHANGED)

# print("zeros = ", np.count_nonzero(np.ravel(gt) == 0))
# print("total pixels = ", len(np.ravel(gt)))
# print("% = ", np.count_nonzero(np.ravel(gt) == 0) / len(np.ravel(gt)) *100)
# box = np.argwhere(mask)

# (ystart, xstart), (ystop, xstop) = box.min(0), box.max(0) + 1

# trim = gt[ystart:ystop, xstart:xstop]

# print(trim)
# cv2.imwrite("./gt_masked.png", filtered_depth)
# pred = pred.astype(np.int64)
# gt = gt.astype(np.int64)
# print(pred)
# print(gt)
# print(pred-gt)
# print(gt-pred)

# epsilon = 1e-3

# abs_relative_error = np.mean(np.abs(pred - gt) / (gt + epsilon))

# print(abs_relative_error)



# for visualization
# gt = cv2.imread("./data/wild6d/test_mixRot_DEPTH/bottle/0001/1/images/0-depth.png", cv2.IMREAD_UNCHANGED)
# gt = gt.astype(np.int64)
# normalized_image = (4000-gt) / 4000.0
# cmap = cm.get_cmap('inferno')
# rgb_image = cmap(normalized_image)
# plt.imsave("./pred_inferno.png", rgb_image)