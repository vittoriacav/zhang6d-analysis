import numpy as np
import pickle
import os
import pandas as pd
import shutil

categories = ["bowl", "camera", "mug", "laptop"]

checkpoint_dir='log'
logname='exp1-kaifeng'
test='TestOrig_MiDaS'
category = [cat for cat in categories if cat in logname]
if not category:
    category = 'bottle'
else:
    category = category[0]

print("category: ", category)

degs_path = "/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/test_{}_mixRot_degs.txt".format(category)

reading_path = os.path.join('/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose', checkpoint_dir, logname, test)
os.makedirs('/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/FORTHESIS/midas_depth/{}'.format(category), exist_ok=True)
writing_path = os.path.join('/home/maria.cavicchioli/vittoria/kaifeng/self-corr-pose/data/FORTHESIS/midas_depth/{}'.format(category))

to_download = ['_000_depth_gt.png']
# to_download = ['_000_3d.png', '_000_bbox.png', '_000_gt.png', '_000_tex.png']

if category in ['bottle', 'camera', 'laptop']:
    all_degs = ['0','10','30', '50']
else:
    all_degs = ['0','10','30','50', '70', '90']

mem = ''
if os.path.exists(degs_path):
    with open(degs_path, 'r') as test_mixrot_degs:
        for line in test_mixrot_degs:
            if line[:4] != mem:
                
                print("\n"+line[:-1])
                mem = line[:4]
                print("changed mem: ", mem)
                parts = line.strip().split('/')
                deg = parts[-1].replace("deg_", "")
                print('found deg: ', deg)
                
                if deg in all_degs:
                    all_degs.remove(deg)
                    start = str(int(mem)-1).zfill(3)
                    for final in to_download:
                        read_img_name = start+final
                        write_img_name = deg+final
                        print("read name: ", read_img_name)
                        print("save name: ", write_img_name)
                        read_file = os.path.join(reading_path, read_img_name)
                        write_file = os.path.join(writing_path, write_img_name)
                        shutil.copyfile(read_file, write_file)
                
                print("len(all_degs)", len(all_degs))
                
                if len(all_degs) == 0:
                    if category in ['bottle', 'camera', 'laptop']:
                        if category == 'bottle':
                            start70 = '074'
                            start90 = '110'
                        else: 
                            start70 = '012'
                            start90 = '008'
                    
                        for final in to_download:
                            read_img_name70 = start70+final
                            write_img_name70 = '70'+final
                            read_file = os.path.join(reading_path, read_img_name70)
                            write_file = os.path.join(writing_path, write_img_name70)
                            shutil.copyfile(read_file, write_file)

                            read_img_name90 = start90+final
                            write_img_name90 = '90'+final
                            read_file = os.path.join(reading_path, read_img_name90)
                            write_file = os.path.join(writing_path, write_img_name90)
                            shutil.copyfile(read_file, write_file)
                    else:
                        break
                            
else:
    print("path {} doesn't exist!".format(degs_path))